import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_ssim
from dataloader import *

Sobel = np.array([[-1,-2,-1],
                  [ 0, 0, 0],
                  [ 1, 2, 1]])
Robert = np.array([[0, 0],
                  [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)

def feature_map_hook(*args, path=None):
    feature_maps = []
    for feature in args:
        feature_maps.append(feature)
    feature_all = torch.cat(feature_maps, dim=1)
    fmap = feature_all.detach().cpu().numpy()[0]
    fmap = np.array(fmap)
    fshape = fmap.shape
    num = fshape[0]
    shape = fshape[1:]
    sample(fmap, figure_size=(2, num//2), img_dim=shape, path=path)
    return fmap

# 已测试本模块没有问题，作用为提取一阶导数算子滤波图（边缘图）
def gradient(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


def gradient_no_abs(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


# 写完才想起来这for在pytorch里好像巨慢，还是不用了吧
def Pyramid_Sample(img, max_scale=8):
    imgs = []
    sample = img
    power = 1
    while 2**power <= max_scale:
        sample = nn.AvgPool2d(2,2)(sample)
        imgs.append(sample)
        power += 1
    return imgs

def Pyramid_Loss(low1,low2,low4,low8,high1,high2,high4,high8):
    loss1 = F.l1_loss(low1, high1)
    loss2 = F.l1_loss(low2, high2)
    loss4 = F.l1_loss(low4, high4)
    loss8 = F.l1_loss(low8, high8)
    loss = (loss1*1 + loss2*2 + loss4*4 + loss8*8) / 15
    return loss


class Decom_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def reflectance_similarity(self, R_low, R_high):
        R_low2 = nn.AvgPool2d(2,2)(R_low)
        R_low4 = nn.AvgPool2d(2,2)(R_low2)
        R_low8 = nn.AvgPool2d(2,2)(R_low4)
        R_high2 = nn.AvgPool2d(2,2)(R_high)
        R_high4 = nn.AvgPool2d(2,2)(R_high2)
        R_high8 = nn.AvgPool2d(2,2)(R_high4)
        # loss = Pyramid_Loss(low1,low2,low4,low8,high1,high2,high4,high8)
        loss1 = F.l1_loss(R_low, R_high)
        loss2 = F.l1_loss(R_low2, R_high2)
        loss4 = F.l1_loss(R_low4, R_high4)
        loss8 = F.l1_loss(R_low8, R_high8)
        loss = (loss1*1 + loss2*2 + loss4*4 + loss8*8) / 15
        return loss
    
    def illumination_smoothness(self, I, L, name='low', hook=-1):
        L_gray = 0.299*L[:,0,:,:] + 0.587*L[:,1,:,:] + 0.114*L[:,2,:,:]
        L_gray = L_gray.unsqueeze(dim=1)
        I_gradient_x = gradient(I, "x")
        L_gradient_x = gradient(L_gray, "x")
        epsilon = 0.01*torch.ones_like(L_gradient_x)
        Denominator_x = torch.max(L_gradient_x, epsilon)
        x_loss = torch.abs(torch.div(I_gradient_x, Denominator_x))
        I_gradient_y = gradient(I, "y")
        L_gradient_y = gradient(L_gray, "y")
        Denominator_y = torch.max(L_gradient_y, epsilon)
        y_loss = torch.abs(torch.div(I_gradient_y, Denominator_y))
        mut_loss = torch.mean(x_loss + y_loss)
        if hook > -1:
            feature_map_hook(I, L_gray, epsilon, I_gradient_x+I_gradient_y, Denominator_x+Denominator_y, 
                            x_loss+y_loss, path=f'./images/samples-features/ilux_smooth_{name}_epoch{hook}.png')
        return mut_loss
    
    def mutual_consistency(self, I_low, I_high, hook=-1):
        low_gradient_x = gradient(I_low, "x")
        high_gradient_x = gradient(I_high, "x")
        M_gradient_x = low_gradient_x + high_gradient_x
        x_loss = M_gradient_x * torch.exp(-10 * M_gradient_x)
        low_gradient_y = gradient(I_low, "y")
        high_gradient_y = gradient(I_high, "y")
        M_gradient_y = low_gradient_y + high_gradient_y
        y_loss = M_gradient_y * torch.exp(-10 * M_gradient_y)
        mutual_loss = torch.mean(x_loss + y_loss) 
        if hook > -1:
            feature_map_hook(I_low, I_high, low_gradient_x+low_gradient_y, high_gradient_x+high_gradient_y, 
                    M_gradient_x + M_gradient_y, x_loss+ y_loss, path=f'./images/samples-features/mutual_consist_epoch{hook}.png')
        return mutual_loss

    def reconstruction_error(self, R_low, R_high, I_low, I_high, L_low, L_high):
        I_low_3 = torch.cat([I_low, I_low, I_low], dim=1)
        I_high_3 = torch.cat([I_high, I_high, I_high], dim=1)
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 -  L_low))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - L_high))

        L_low4 = nn.AvgPool2d(4,4)(L_low)
        L_low8 = nn.AvgPool2d(2,2)(L_low4)
        L_high4 = nn.AvgPool2d(4,4)(L_high)
        L_high8 = nn.AvgPool2d(2,2)(L_high4)

        R_low4 = nn.AvgPool2d(4,4)(R_low)
        R_low8 = nn.AvgPool2d(2,2)(R_low4)
        R_high4 = nn.AvgPool2d(4,4)(R_high)
        R_high8 = nn.AvgPool2d(2,2)(R_high4)

        I_low4 = nn.AvgPool2d(4,4)(I_low_3)
        I_low8 = nn.AvgPool2d(2,2)(I_low4)
        I_high4 = nn.AvgPool2d(4,4)(I_high_3)
        I_high8 = nn.AvgPool2d(2,2)(I_high4)

        loss4_h2l =  F.l1_loss(R_high4 * I_low4, L_low4)
        loss8_h2l =  F.l1_loss(R_high8 * I_low8, L_low8)
        loss4_l2h =  F.l1_loss(R_low4 * I_high4, L_high4)
        loss8_l2h =  F.l1_loss(R_low8 * I_high8, L_high8)
        recon_loss_cross = (loss4_h2l + loss4_l2h + 2*loss8_h2l + 2*loss8_l2h) / 3
        
        return recon_loss_high + recon_loss_low + recon_loss_cross*0.01

    def forward(self, R_low, R_high, I_low, I_high, L_low, L_high, hook=-1):
        #network output
        recon_loss = self.reconstruction_error(R_low, R_high, I_low, I_high, L_low, L_high)
        equal_R_loss = self.reflectance_similarity(R_low, R_high)
        i_mutual_loss = self.mutual_consistency(I_low, I_high, hook=hook)
        # ilux_smooth_loss = self.illumination_smoothness(I_low, L_low, hook=hook) + \
        #             self.illumination_smoothness(I_high, L_high, name='high', hook=hook) 

        decom_loss = recon_loss + 0.01*equal_R_loss + 0.1 * i_mutual_loss# + 0.015 * ilux_smooth_loss

        return decom_loss


class Illum_Custom_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def grad_loss(self, low, high):
        x_loss = F.l1_loss(gradient_no_abs(low, 'x'), gradient_no_abs(high, 'x'))
        y_loss = F.l1_loss(gradient_no_abs(low, 'y'), gradient_no_abs(high, 'y'))
        grad_loss_all = x_loss + y_loss
        return grad_loss_all

    def gamma_loss(self, I_standard, I_high):
        loss = F.l1_loss(I_high, I_standard)
        return loss

    def forward(self, I_low, I_high, I_standard):
        loss_gamma = self.gamma_loss(I_standard, I_high)
        loss_grad = self.grad_loss(I_low, I_high)
        loss_recon = F.l1_loss(I_low, I_high)
        loss_adjust = loss_recon + loss_grad + loss_gamma*0.0001
        return loss_adjust


class Restore_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_loss = pytorch_ssim.SSIM()
    
    def grad_loss(self, low, high, hook=-1):
        x_loss = F.mse_loss(gradient_no_abs(low, 'x'), gradient_no_abs(high, 'x'))
        y_loss = F.mse_loss(gradient_no_abs(low, 'y'), gradient_no_abs(high, 'y'))
        grad_loss_all = x_loss + y_loss
        return grad_loss_all

    def loss(self, low, high):
        loss_grad = self.grad_loss(low, high)
        loss_recon = F.l1_loss(low, high)
        loss_ssim = 1-self.ssim_loss(low, high)
        return loss_recon + loss_ssim + loss_grad

    def forward(self, R_low, R_high, L2,L4,L8,hook=-1):
        H2,H4,H8 = Pyramid_Sample(R_high, max_scale=8)
        loss_restore = self.loss(R_low, R_high)+self.loss(L2,H2)/2+self.loss(L4,H4)/4+self.loss(L8,H8)/8
        # loss_grad = self.grad_loss(R_low, R_high, hook=hook)
        # loss_recon = F.l1_loss(R_low, R_high)
        # loss_ssim = 1-self.ssim_loss(R_low, R_high)
        # loss_restore = loss_recon + loss_ssim #+ loss_grad
        return loss_restore


class Unet_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_loss = pytorch_ssim.SSIM()
    
    def gradient_loss(self, low, high):
        low_gradient_x = gradient(low, "x")
        high_gradient_x = gradient(high, "x")
        low_gradient_y = gradient(low, "y")
        high_gradient_y = gradient(high, "y")
        x_loss = F.l1_loss(low_gradient_x, high_gradient_x)
        y_loss = F.l1_loss(low_gradient_y, high_gradient_y)
        gradient_loss = torch.mean(x_loss + y_loss) 
        return gradient_loss

    def forward(self, low, high, hook=-1):
        # low = low[:,:-1,:,:]
        high = high[:,:-1,:,:]
        loss_recon = F.l1_loss(low, high)
        # loss_grad = self.gradient_loss(low, high)
        loss_ssim = 1-self.ssim_loss(low, high)
        loss_restore = loss_recon + loss_ssim
        return loss_restore


if __name__ == "__main__":
    from dataloader import *
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    root_path_train = r'H:\datasets\Low-Light Dataset\LOLdataset_decom\eval15'
    list_path_train = build_LOLDataset_Decom_list_txt(root_path_train)
    Batch_size = 1
    log("Buliding LOL Dataset...")
    dst_test = LOLDataset_Decom(root_path_train, list_path_train, to_RAM=True, training=False)
    # But when we are training a model, the mean should have another value
    testloader = DataLoader(dst_test, batch_size = Batch_size)
    for i, data in enumerate(testloader):
        L_low, I_low, L_high, I_high, name = data
        L_low_2 = nn.AvgPool2d((2,2))(L_low)
        L_low_4 = nn.AvgPool2d((2,2))(L_low_2)
        L_low_8 = nn.AvgPool2d((2,2))(L_low_4)
        L_high_2 = nn.AvgPool2d((2,2))(L_high)
        L_high_4 = nn.AvgPool2d((2,2))(L_high_2)
        L_high_8 = nn.AvgPool2d((2,2))(L_high_4)
        imgs = torch.cat([L_low, L_high], dim=1)
        imgs_2 = torch.cat([L_low_2, L_high_2], dim=1)
        imgs_4 = torch.cat([L_low_4, L_high_4], dim=1)
        imgs_8 = torch.cat([L_low_8, L_high_8], dim=1)
        # L_gradient_low = gradient_no_abs(L_high, "x", device='cpu', kernel='sobel') + \
        #                  gradient_no_abs(L_high, "y", device='cpu', kernel='sobel')
        # L_gradient_high = gradient_no_abs(L_low, "x", device='cpu', kernel='sobel') + \
        #                   gradient_no_abs(L_low, "y", device='cpu', kernel='sobel')
        # loss = torch.abs(L_gradient_low - L_gradient_high)
        # imgs = torch.cat([L_gradient_low, L_gradient_high, loss], dim=1)
        log(name)
        img = imgs[0].numpy()
        sample(img, figure_size=(1,2), img_dim=img.shape[-2:])
        img = imgs_2[0].numpy()
        sample(img, figure_size=(1,2), img_dim=img.shape[-2:])
        img = imgs_4[0].numpy()
        sample(img, figure_size=(1,2), img_dim=img.shape[-2:])
        img = imgs_8[0].numpy()
        sample(img, figure_size=(1,2), img_dim=img.shape[-2:])