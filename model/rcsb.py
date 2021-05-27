import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CSFBUnit(nn.Module):
    def __init__(self,in_channel):
        super(CSFBUnit,self).__init__()
        self.conv_head_ctr = nn.Sequential(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1, bias=True),
                                           nn.BatchNorm2d(in_channel),
                                           nn.LeakyReLU(inplace=True))
        self.conv_head_sal = nn.Sequential(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1, bias=True),
                                           nn.BatchNorm2d(in_channel),
                                           nn.LeakyReLU(inplace=True))
        self.merge_sal = nn.Sequential(nn.Conv2d(in_channel*2,in_channel,kernel_size=1,padding=0, bias=True),
                                       nn.GroupNorm(in_channel//2, in_channel),
                                       nn.LeakyReLU(inplace=True))
        self.merge_ctr = nn.Sequential(nn.Conv2d(in_channel*2,in_channel,kernel_size=1,padding=0, bias=True),
                                       nn.GroupNorm(in_channel//2, in_channel),
                                       nn.LeakyReLU(inplace=True))
        self.conv_tail_ctr = nn.Sequential(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1, bias=True),
                                           nn.BatchNorm2d(in_channel),
                                           nn.LeakyReLU(inplace=True))
        self.conv_tail_sal = nn.Sequential(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1, bias=True),
                                           nn.BatchNorm2d(in_channel),
                                           nn.LeakyReLU(inplace=True))
        
    def forward(self, x):
        ctr, sal = x
        ctr = self.conv_head_ctr(ctr)
        sal = self.conv_head_sal(sal)
        
        ctr_n_sal = torch.cat([ctr, sal], dim=1)
        ctr_sal = self.merge_sal(ctr_n_sal)
        sal_ctr = self.merge_ctr(ctr_n_sal)
        
        ctr = self.conv_tail_ctr(ctr_sal)
        sal = self.conv_tail_sal(sal_ctr)
        
        return ctr, sal
    

class CSFBBlock(nn.Module):
    def __init__(self,in_channel, R):
        super(CSFBBlock,self).__init__()
        self.fbunit = CSFBUnit(in_channel=in_channel)
        self.tail_ctr = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True),
                                      nn.BatchNorm2d(in_channel),
                                      nn.LeakyReLU(inplace=True))
        self.tail_sal = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True),
                                      nn.BatchNorm2d(in_channel),
                                      nn.LeakyReLU(inplace=True))
        self.R = R

    def forward(self, x):
        ctr, sal = x
        h0_ctr = ctr
        h0_sal = sal
        
        for _ in range(self.R):
            ctr, sal = self.fbunit([ctr, sal])
            ctr = ctr + h0_ctr
            sal = sal + h0_sal
            
        ctr = self.tail_ctr(ctr)
        sal = self.tail_sal(sal)

        ctr = ctr + h0_ctr
        sal = sal + h0_sal
        
        return ctr, sal
    

class ChannelAdapter(nn.Module):
    def __init__(self,num_features, reduction=4, reduce_to=64):
        super(ChannelAdapter,self).__init__()
        self.n = reduction
        self.reduce = num_features>64
        self.conv = nn.Sequential(nn.Conv2d(num_features//self.n if self.reduce else reduce_to,
                                            reduce_to,kernel_size=3,padding=1, bias=True),
                                  nn.LeakyReLU(inplace=True))
        
    def forward(self, x):
        # reduce dimension
        if self.reduce:
            batch, c, w, h = x.size()
            x = x.view(batch, -1, self.n, w, h)
            x = torch.max(x, dim=2).values
        # conv
        xn = self.conv(x)
        return xn

class MapAdapter(nn.Module):
    def __init__(self,num_features):
        super(MapAdapter,self).__init__()
        self.conv_ctr = nn.Conv2d(num_features, 1, kernel_size=1,padding=0, bias=True)
        self.conv_sal = nn.Conv2d(num_features, 1, kernel_size=1,padding=0, bias=True)
        self.conv_end = nn.Conv2d(2, num_features, kernel_size=3,padding=1, bias=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sal_scale = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.ctr_scale = nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, ctr, sal):
        pred_ctr = self.conv_ctr(ctr) * self.ctr_scale
        pred_sal = self.conv_sal(sal) * self.sal_scale
        
        merge = torch.cat([pred_ctr, pred_sal], dim=1)
        merge = torch.sigmoid(merge)

        stage_feature = self.conv_end(merge)
        stage_feature = self.relu(stage_feature)

        return pred_ctr, pred_sal, stage_feature


class MergeAdapter(nn.Module):
    def __init__(self, in_features, out_features):
        super(MergeAdapter, self).__init__()
        self.merge_head = nn.Sequential(nn.Conv2d(in_features,out_features,kernel_size=1,padding=0, bias=True),
                                        nn.BatchNorm2d(out_features),
                                        nn.LeakyReLU(inplace=True))

    def forward(self, x):
        out = self.merge_head(x)

        return out


class Net(nn.Module):
    def __init__(self, opt):
        super(Net,self).__init__()
        
        self.output = dict()
        num_features = opt.num_features
        G = opt.G
        R = opt.R

        resnet50 = models.resnet50(pretrained=True)
        head = list(resnet50.children())[:3]
        self.backbone0 = nn.Sequential(*head)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.backbone1 = resnet50.layer1
        self.backbone2 = resnet50.layer2
        self.backbone3 = resnet50.layer3
        self.backbone4 = resnet50.layer4

        self.CSFB0 = nn.Sequential(*[CSFBBlock(num_features, R=R) for i in range(3*G)])
        self.CSFB1 = nn.Sequential(*[CSFBBlock(num_features, R=R) for i in range(3*G)])
        self.CSFB2 = nn.Sequential(*[CSFBBlock(num_features, R=R) for i in range(2*G)])
        self.CSFB3 = nn.Sequential(*[CSFBBlock(num_features, R=R) for i in range(2*G)])
        self.CSFB4 = nn.Sequential(*[CSFBBlock(num_features, R=R) for i in range(1*G)])

        self.CSFB_end = nn.Sequential(*[CSFBBlock(num_features, R=R) for i in range(5 * G)])
        self.final_sal = nn.Conv2d(num_features, 1, 3, padding=1, bias=True)
        self.final_ctr = nn.Conv2d(num_features, 1, 3, padding=1, bias=True)
        self.map_gen = nn.ModuleList([MapAdapter(num_features) for i in range(5)])

        self.merge = nn.ModuleList([MergeAdapter(in_features=num_features*2,
                                                 out_features=num_features) for i in range(4)])

        self.merge_end = MergeAdapter(in_features=5,
                                      out_features=num_features)

        self.adapter = nn.ModuleList([ChannelAdapter(num_features=channel) for channel in [64, 256, 512, 1024, 2048]])
        self.sal_final_scale = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.ctr_final_scale = nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, x):
        
        out0 = self.backbone0(x)
        out0_p = self.pool(out0)
        out1 = self.backbone1(out0_p)
        out2 = self.backbone2(out1)
        out3 = self.backbone3(out2)
        out4 = self.backbone4(out3)
        
        A0 = self.adapter[0](out0)
        A1 = self.adapter[1](out1)
        A2 = self.adapter[2](out2)
        A3 = self.adapter[3](out3)
        A4 = self.adapter[4](out4)

        # BLK 4
        C4, S4 = self.CSFB4([A4, A4])
        C4_map, S4_map, S4M = self.map_gen[0](C4, S4)

        S4M_x2 = F.interpolate(S4M, scale_factor=2, mode='bilinear', align_corners=False)
        # merge
        M4_3 = torch.cat([S4M_x2, A3], dim=1)
        M4_3 = self.merge[0](M4_3)
        # BLK 3
        C3, S3 = self.CSFB3([M4_3, M4_3])
        C3_map, S3_map, S3M = self.map_gen[1](C3, S3)

        S3M_x2 = F.interpolate(S3M, scale_factor=2, mode='bilinear', align_corners=False)
        # merge
        M3_2 = torch.cat([S3M_x2, A2], dim=1)
        M3_2 = self.merge[1](M3_2)
        # BLK 2
        C2, S2 = self.CSFB2([M3_2, M3_2])
        C2_map, S2_map, S2M = self.map_gen[2](C2, S2)

        S2M_x2 = F.interpolate(S2M, scale_factor=2, mode='bilinear', align_corners=False)
        # merge
        M2_1 = torch.cat([S2M_x2, A1], dim=1)
        M2_1 = self.merge[2](M2_1)
        # BLK 1
        C1, S1 = self.CSFB2([M2_1, M2_1])
        C1_map, S1_map, S1M = self.map_gen[3](C1, S1)

        S1M_x2 = F.interpolate(S1M, scale_factor=2, mode='bilinear', align_corners=False)
        # merge
        M0_1 = torch.cat([S1M_x2, A0], dim=1)
        M0_1 = self.merge[3](M0_1)
        # BLK 0
        C0, S0 = self.CSFB0([M0_1, M0_1])
        C0_map, S0_map, S0M = self.map_gen[4](C0, S0)
        
        # ref
        S0Map_x2 = F.interpolate(S0_map, scale_factor=2, mode='bilinear', align_corners=False)
        C0Map_x2 = F.interpolate(C0_map, scale_factor=2, mode='bilinear', align_corners=False)
        SFM = torch.cat([S0Map_x2, C0Map_x2, x], dim=1)
        M0_end = self.merge_end(SFM)
        ctr_end, sal_end = self.CSFB_end([M0_end, M0_end])
        
        sal_pred = self.final_sal(sal_end) * self.sal_final_scale
        ctr_pred = self.final_ctr(ctr_end) * self.ctr_final_scale
        
        # map size: small -> big
        self.output['sal'] = [S4_map, S3_map, S2_map, S1_map, S0_map, sal_pred]
        self.output['ctr'] = [C4_map, C3_map, C2_map, C1_map, C0_map, ctr_pred]
        
        return self.output
