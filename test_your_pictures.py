import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import yaml
import sys
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from torchvision import transforms
from torchsummary import summary
from base_trainer import BaseTrainer
from losses import *
from models import *
from base_parser import BaseParser
from dataloader import *

class KinD_Player(BaseTrainer):
    def __init__(self, model, dataloader_test, plot_more=False):
        self.dataloader_test = dataloader_test
        self.model = model
        self.plot_more = plot_more
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)

    @no_grad
    def test(self, ratio=1.4, plot_dir='./images/samples-KinD'):
        self.model.eval()
        self.model.to(device=self.device)
        for L_low_tensor, name in self.dataloader_test:
            L_low = L_low_tensor.to(self.device)
            with torch.no_grad():
                if self.plot_more:
                    R_low, I_low = self.model.decom_net(L_low)
                # # 连续亮度过渡
                # R_low, I_low = self.model.decom_net(L_low)
                # I_final, I_standard = self.model.illum_net(I_low, ratio)
                # R_final = self.model.restore_net(R_low, I_low)
                # I_att = torch.clamp(F.sigmoid((.5-I_low)*10)/0.99330, min=0.2, max=1.0)
                # R_final = R_low + (R_final-R_low) * torch.cat([I_att, I_att, I_att], dim=1)

                # for step in range(5):
                #     I_step = 0.25*(step*I_final + (4-step)*I_standard)
                #     I_final_3 = torch.cat([I_step, I_step, I_step], dim=1)
                #     output_final = I_final_3 * R_final
                #     output_final_np = output_final.detach().cpu().numpy()[0]
                #     L_low_np = L_low_tensor.numpy()[0]
                #     # Only plot result 
                #     filepath = os.path.join(plot_dir, f'{name[0]}_{step}.png')
                #     split_point = [0, 3]
                #     img_dim = L_low_np.shape[1:]
                #     sample(output_final_np, split=split_point, figure_size=(1, 1), 
                #                 img_dim=img_dim, path=filepath)
                R_final, I_final, output_final = self.model(L_low, ratio, limit_highlight=True)

            output_final_np = output_final.detach().cpu().numpy()[0]
            L_low_np = L_low_tensor.numpy()[0]
            # Only plot result 
            filepath = os.path.join(plot_dir, f'{name[0]}.png')
            split_point = [0, 3]
            img_dim = L_low_np.shape[1:]
            sample(output_final_np, split=split_point, figure_size=(1, 1), 
                        img_dim=img_dim, path=filepath)

            if self.plot_more:
                R_final_np = R_final.detach().cpu().numpy()[0]
                I_final_np = I_final.detach().cpu().numpy()[0]
                R_low_np = R_low.detach().cpu().numpy()[0]
                I_low_np = I_low.detach().cpu().numpy()[0]
                
                sample_imgs = np.concatenate( (R_low_np, I_low_np, L_low_np,
                                            R_final_np, I_final_np, output_final_np), axis=0 )
                filepath = os.path.join(plot_dir, f'{name[0]}_extra.png')
                split_point = [0, 3, 4, 7, 10, 11, 14]
                img_dim = L_low_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(2, 3), 
                            img_dim=img_dim, path=filepath)
                        
            
class TestParser(BaseParser):
    def parse(self):
        self.parser.add_argument("-p", "--plot_more", default=True,
                                help="Plot intermediate variables. such as R_images and I_images")
        self.parser.add_argument("-c", "--checkpoint", default="./weights/", 
                                help="Path of checkpoints")
        self.parser.add_argument("-i", "--input_dir", default="./images/inputs-light/", 
                                help="Path of input pictures")
        self.parser.add_argument("-o", "--output_dir", default="./images/outputs-illum-custom-light/", 
                                help="Path of output pictures")
        self.parser.add_argument("-r", "--ratio", default=1.5, help="Target brightness")
        # self.parser.add_argument("-u", "--use_gpu", default=True, 
        #                         help="If you want to use GPU to accelerate")
        return self.parser.parse_args()


if __name__ == "__main__":
    model = KinD(use_MaskMul=True)
    parser = TestParser()
    args = parser.parse()

    input_dir = args.input_dir
    output_dir = args.output_dir
    plot_more = args.plot_more
    checkpoint = args.checkpoint
    decom_net_dir = os.path.join(checkpoint, "decom_net_normal.pth")
    restore_net_dir = os.path.join(checkpoint, "restore_GAN_mask.pth")
    illum_net_dir = os.path.join(checkpoint, "illum_net_custom_final.pth")
    
    model.decom_net = load_weights(model.decom_net, path=decom_net_dir)
    log('Model loaded from decom_net.pth')
    model.restore_net = load_weights(model.restore_net, path=restore_net_dir)
    log('Model loaded from restore_net.pth')
    model.illum_net = load_weights(model.illum_net, path=illum_net_dir)
    log('Model loaded from illum_net.pth')

    log("Buliding Dataset...")
    dst = CustomDataset(input_dir)
    log(f"There are {len(dst)} images in the input direction...")
    dataloader = DataLoader(dst, batch_size=1)

    KinD = KinD_Player(model, dataloader, plot_more=plot_more)
    
    KinD.test(plot_dir=output_dir, ratio=args.ratio)