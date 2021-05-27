import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K


class ConfidentLoss:
    def __init__(self, lmbd=1):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.weight = [0.1, 0.3, 0.5, 0.7, 0.9, 1.5]
        self.lmbda = float(int(lmbd)/10)

    def gen_ctr_torch(self, IMG, kernel_size=5):
        device = IMG.device
        kernel = torch.ones((kernel_size, kernel_size)).to(device)
        C = K.morphology.dilation(IMG, kernel) - K.morphology.erosion(IMG, kernel)
        return C

    def confine(self, source_pred, source_y, target_pred, target_y):
        source_p = torch.sigmoid(source_pred)
        map_source = torch.where(source_p > source_y, source_p, source_y)
        w_source = 4 * map_source + 1
        w_source = w_source.detach()
        loss = (self.bce(target_pred, target_y) * w_source).mean()

        return loss

    def weighted_iou(self, pred, gt):
        weit = 1+4*torch.abs(F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt)
        y = torch.sigmoid(pred)
        AND = ((y*gt)*weit).sum(dim=[2,3])
        OR = ((y+gt)*weit).sum(dim=[2,3])
        wiou = 1-(AND+1)/(OR-AND+1)
        return wiou.mean()

    def confident_loss(self, pred, gt, beta=2):
        y = torch.sigmoid(pred)
        weight = beta*y*(1-y)
        weight = weight.detach()
        p1 = (self.bce(pred, gt) * weight).mean()
        p2 = self.lmbda*beta*(y*(1-y)).mean()
        return p1 + p2
            
    def get_value(self, X, sal_gt, ctr_gt):
        sal_loss, ctr_loss = 0, 0
        count = 0

        for sal_pred, ctr_pred, wght in zip(X['sal'], X['ctr'], self.weight):

            scale = int(sal_gt.size(-1) / sal_pred.size(-1))
            if scale > 1:
                sal_y = F.avg_pool2d(sal_gt, kernel_size=scale, stride=scale).gt(0.5).float()
                ctr_y = self.gen_ctr_torch(sal_y, kernel_size=3).gt(0.5).float()
            else:
                sal_y = sal_gt.gt(0.5).float()
                ctr_y = ctr_gt.gt(0.5).float()

            if count != len(self.weight)-1:
                stage_sal_loss = wght * self.confine(source_pred=ctr_pred, source_y=ctr_y, 
                                                     target_pred=sal_pred,target_y=sal_y)
                stage_ctr_loss = wght * self.confine(source_pred=sal_pred, source_y=sal_y, 
                                                     target_pred=ctr_pred,target_y=ctr_y)
                
                # iou
                stage_sal_loss += wght * self.weighted_iou(sal_pred, sal_y)

            else:
                # last stage use confident loss
                stage_sal_loss += wght * self.confident_loss(sal_pred, sal_y, beta=2)
                stage_ctr_loss += wght * self.confident_loss(ctr_pred, ctr_y, beta=2)
                
            sal_loss += stage_sal_loss
            ctr_loss += stage_ctr_loss

            count += 1

        final_loss = sal_loss + ctr_loss
        return final_loss
