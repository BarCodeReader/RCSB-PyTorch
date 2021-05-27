import os
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import generate_loader
from tqdm import tqdm
from utils import calculate_mae


class Tester():
    def __init__(self, module, opt):
        self.opt = opt

        self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt)
        self.net = self.net.to(self.dev)
        
        msg = "# params:{}\n".format(
            sum(map(lambda x: x.numel(), self.net.parameters())))
        print(msg)

        self.test_loader = generate_loader("test", opt)

    @torch.no_grad()
    def evaluate(self, path):
        opt = self.opt

        try:
            print('loading model from: {}'.format(path))
            self.load(path)
        except Exception as e:
            print(e)

        self.net.eval()

        if opt.save_result:
            save_root = os.path.join(opt.save_root, opt.test_dataset)
            os.makedirs(save_root, exist_ok=True)

        mae = 0

        for i, inputs in enumerate(tqdm(self.test_loader)):
            MASK = inputs[0].to(self.dev)
            IMG = inputs[1].to(self.dev)
            NAME = inputs[2][0]

            b, c, h, w = MASK.shape

            SOD = self.net(IMG)

            MASK = MASK.squeeze().detach().cpu().numpy()
            pred_sal, pred_ctr = SOD['sal'][-1], SOD['ctr'][-1]
            pred_sal = F.interpolate(pred_sal, (h, w), mode='bilinear', align_corners=False)
            pred_ctr = F.interpolate(pred_ctr, (h, w), mode='bilinear', align_corners=False)

            pred_sal = torch.sigmoid(pred_sal).squeeze().detach().cpu().numpy()
            pred_ctr = torch.sigmoid(pred_ctr).squeeze().detach().cpu().numpy()
            pred_sal_img = (pred_sal * 255.).astype('uint8')
            pred_ctr_img = (pred_ctr * 255.).astype('uint8')

            if opt.save_result:
                save_path_sal = os.path.join(save_root, "{}_sal.png".format(NAME))
                save_path_ctr = os.path.join(save_root, "{}_ctr.png".format(NAME))
                io.imsave(save_path_sal, pred_sal_img)
                io.imsave(save_path_ctr, pred_ctr_img)
                if opt.save_all:
                    for idx, ctr in enumerate(SOD['ctr'][:-1]):
                        ctr_path = os.path.join(save_root, "{}_ctr_{}.png".format(NAME, idx))
                        ctr_img = torch.sigmoid(ctr).squeeze().detach().cpu().numpy()
                        ctr_img = (ctr_img * 255).astype('uint8')
                        io.imsave(ctr_path, ctr_img)
                    for idx, sal in enumerate(SOD['sal'][:-1]):
                        sal_path = os.path.join(save_root, "{}_sal_{}.png".format(NAME, idx))
                        sal_img = torch.sigmoid(sal).squeeze().detach().cpu().numpy()
                        sal_img = (sal_img * 255).astype('uint8')
                        io.imsave(sal_path, sal_img)

            mae += calculate_mae(MASK, pred_sal)
            
        return mae/len(self.test_loader)

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        return
