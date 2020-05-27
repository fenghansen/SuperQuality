import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_layers import *
from utils import *

class DecomNet(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.conv_input = Conv2D(3, filters)
        # top path build Reflectance map
        self.maxpool_r1 = MaxPooling2D()
        self.conv_r1 = Conv2D(filters, filters*2)
        self.maxpool_r2 = MaxPooling2D()
        self.conv_r2 = Conv2D(filters*2, filters*4)
        self.deconv_r1 = ConvTranspose2D(filters*4, filters*2)
        self.concat_r1 = Concat()
        self.conv_r3 = Conv2D(filters*4, filters*2)
        self.deconv_r2 = ConvTranspose2D(filters*2, filters)
        self.concat_r2 = Concat()
        self.conv_r4 = Conv2D(filters*2, filters)
        self.conv_r5 = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self.R_out = nn.Sigmoid()
        # bottom path build Illumination map
        self.conv_i1 = Conv2D(filters, filters)
        self.concat_i1 = Concat()
        self.conv_i2 = nn.Conv2d(filters*2, 1, kernel_size=3, padding=1)
        self.I_out = nn.Sigmoid()

    def forward(self, x):
        conv_input = self.conv_input(x)
        # build Reflectance map
        maxpool_r1 = self.maxpool_r1(conv_input)
        conv_r1 = self.conv_r1(maxpool_r1)
        maxpool_r2 = self.maxpool_r2(conv_r1)
        conv_r2 = self.conv_r2(maxpool_r2)
        deconv_r1 = self.deconv_r1(conv_r2)
        concat_r1 = self.concat_r1(conv_r1, deconv_r1)
        conv_r3 = self.conv_r3(concat_r1)
        deconv_r2 = self.deconv_r2(conv_r3)
        concat_r2 = self.concat_r2(conv_input, deconv_r2)
        conv_r4 = self.conv_r4(concat_r2)
        conv_r5 = self.conv_r5(conv_r4)
        R_out = self.R_out(conv_r5)
        
        # build Illumination map
        conv_i1 = self.conv_i1(conv_input)
        concat_i1 = self.concat_i1(conv_r4, conv_i1)
        conv_i2 = self.conv_i2(concat_i1)
        I_out = self.I_out(conv_i2)

        return R_out, I_out

class Standard_Illum(nn.Module):
    def __init__(self, w=0.5, sigma=2.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w, requires_grad=True, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, requires_grad=True, dtype=torch.float32))

        self.Gauss = torch.as_tensor(
                        np.array([[0.0947416, 0.118318, 0.0947416],
                                [ 0.118318, 0.147761, 0.118318],
                                [0.0947416, 0.118318, 0.0947416]]).astype(np.float32)
                        )
        self.Gauss_kernel = self.Gauss.expand(1, 1, 3, 3)

    def set_parameter(self, w=None, sigma=None):
        if w is None:
            self.w.requires_grad = False
            self.sigma.requires_grad = False
        else:
            self.w.data.fill_(w)
            self.sigma.data.fill_(sigma)
            self.w.requires_grad = True
            self.sigma.requires_grad = True
    
    def get_parameter(self):
        if self.w.device.type == 'cuda':
            w = self.w.detach().cpu().numpy()
            sigma = self.sigma.detach().cpu().numpy()
        else:
            w = self.w.numpy()
            sigma = self.sigma.numpy()
        return w, sigma

    def forward(self, I, ratio):
        # if blur: # low light image have much noisy 
        #     I = torch.nn.functional.conv2d(I, weight=self.Gauss_kernel, padding=1)
        # I = torch.log(I + 1.)
        I_mean = torch.mean(I, dim=[2, 3], keepdim=True)
        I_std = torch.std(I, dim=[2, 3], keepdim=True)
        I_min = I_mean - self.sigma * I_std
        I_max = I_mean + self.sigma * I_std
        tmp_low, _ = torch.min(I,dim=2,keepdim=True)
        I_below, _ = torch.min(tmp_low,dim=3,keepdim=True)
        tmp_high,_ = torch.max(I,dim=2,keepdim=True)
        I_top, _  = torch.max(tmp_high,dim=3,keepdim=True)
        I_range_min = torch.max(I_min, I_below)
        I_range_max = torch.min(I_max, I_top)
        I_range = I_range_max - I_range_min
        I_out = torch.clamp((I - I_range_min) / I_range, min=0.0, max=1.0)
        # Transfer to gamma correction, center intensity is w
        I_out = I_out ** (1 / (self.w * ratio) - 1)
        return I_out

class IllumNet(nn.Module):
    def __init__(self, filters=16, w=0.5, sigma=2.0):
        super().__init__()
        self.concat_input = Concat()
        self.I_standard = Standard_Illum(w=w, sigma=sigma)
        self.I_standard.set_parameter() # 先锁住

        # bottom path build Illumination map
        self.conv_input = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.res_block1 = ResConv(filters, filters)
        self.res_block2 = ResConv(filters, filters)
        self.res_block3 = ResConv(filters, filters)
        self.conv_out = conv1x1(filters, 1)

        self.I_out = nn.Sigmoid()

    def forward(self, I, ratio):
        with torch.no_grad():
            I_standard = self.I_standard(I, ratio)
        # concat_input = torch.cat([I, I_standard], dim=1)
        # build Illumination map
        # conv_input = self.conv_input(concat_input)
        conv_input = self.conv_input(I_standard)

        res_block1 = self.res_block1(conv_input)
        res_block2 = self.res_block2(res_block1)
        res_block3 = self.res_block3(res_block2)
        res_out = res_block3 + conv_input
        conv_out = self.conv_out(res_out)

        I_out = conv_out# + I_standard

        return I_out, I_standard

class IllumNet_Custom(nn.Module):
    def __init__(self, filters=16, w=0.5, sigma=2.0):
        super().__init__()
        self.concat_input = Concat()
        self.I_standard = Standard_Illum(w=w, sigma=sigma)
        self.I_standard.set_parameter() # 先锁住

        # bottom path build Illumination map
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.res_block1 = ResConv(filters, filters)
        self.ca1 = ChannelAttention(filters*2)
        self.bottleneck1 = conv1x1(filters*2, filters)
        self.res_block2 = ResConv(filters, filters)
        self.ca2 = ChannelAttention(filters*2)
        self.bottleneck2 = conv1x1(filters*2, filters)
        self.res_block3 = ResConv(filters, filters)
        self.ca3 = ChannelAttention(filters*2)
        self.bottleneck3 = conv1x1(filters*2, filters)
        self.conv_out = nn.Conv2d(filters*1, 1, kernel_size=1, padding=0)

        self.fusion = conv1x1(2,1,bias=False)
        self.I_out = nn.Sigmoid()

    def forward(self, I, ratio):
        I_Att = 1-I
        with torch.no_grad():
            I_standard = self.I_standard(I, ratio)
        concat_input = torch.cat([I, I_standard, I_Att], dim=1)
        # build Illumination map
        conv_input = self.conv_input(concat_input)

        res_block1 = self.res_block1(conv_input)
        res_concat1 = torch.cat([res_block1, MaskMul(1)(res_block1, I_Att)], dim=1)
        ca1 = res_concat1 * self.ca1(res_concat1)
        bottleneck1 = self.bottleneck1(ca1)

        res_block2 = self.res_block2(bottleneck1)
        res_concat2 = torch.cat([res_block2, MaskMul(1)(res_block2, I_Att)], dim=1)
        ca2 = res_concat2 * self.ca2(res_concat2)
        bottleneck2 = self.bottleneck2(ca2)

        res_block3 = self.res_block3(bottleneck2)
        res_out = res_block3 + conv_input

        res_concat3 = torch.cat([res_block3, MaskMul(1)(res_out, I_Att)], dim=1)
        ca3 = res_concat3 * self.ca3(res_concat3)
        bottleneck3 = self.bottleneck3(ca3)

        conv_out = self.conv_out(bottleneck3)
        fusion = self.fusion(torch.cat([conv_out, I_standard],dim=1))
        I_out = self.I_out(fusion)

        return I_out, I_standard

class Illum_D(nn.Module):
    def __init__(self, filters=16, activation='lrelu'):
        super().__init__()
        if activation == 'relu':
            self.relu = nn.ReLU()
        else:
            self.relu = nn.LeakyReLU(0.2)
        self.conv1 = DoubleConv(1, filters, stride=2)
        self.conv2 = DoubleConv(filters*1, filters*2, stride=2)
        self.conv3 = DoubleConv(filters*2, filters*4, stride=2)
        self.conv4 = DoubleConv(filters*4, filters*8, stride=2)
        self.conv5 = DoubleConv(filters*8, filters*8) # [b, 512, 1/16, 1/16]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dense = conv1x1(filters*8, filters*8)
        self.dropout = nn.Dropout(0.4)
        self.out = conv1x1(filters*8, 1)
    
    def forward(self, R):
        conv1 = self.conv1(R)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        dense = self.relu(self.dense(self.pool(conv5)))
        d = self.dropout(dense)
        out = self.out(d)
        return out

class RestoreNet_Unet(nn.Module):
    def __init__(self, filters=32, activation='lrelu', use_MaskMul=False):
        super().__init__()
        self.use_MaskMul = use_MaskMul

        self.conv1 = DoubleConv(5, filters)
        self.pool1 = MaxPooling2D()
        
        self.conv2 = DoubleConv(filters, filters*2)
        self.pool2 = MaxPooling2D()
        
        self.conv3 = DoubleConv(filters*2, filters*4)
        self.pool3 = MaxPooling2D()
        
        self.conv4 = DoubleConv(filters*4, filters*8)
        self.pool4 = MaxPooling2D()
        
        self.conv5 = DoubleConv(filters*8, filters*16)
        self.dropout = nn.Dropout2d(0.5)

        self.upv6 = Up(filters*16, filters*8, mode='biliner')
        self.concat6 = Concat()
        self.ca6 = ChannelAttention(filters*16)
        self.conv6 = DoubleConv(filters*16, filters*8)
        
        self.upv7 = Up(filters*8, filters*4, mode='biliner')
        self.concat7 = Concat()
        self.ca7 = ChannelAttention(filters*8)
        self.conv7 = DoubleConv(filters*8, filters*4)
        
        self.upv8 = Up(filters*4, filters*2, mode='biliner')
        self.concat8 = Concat()
        self.ca8 = ChannelAttention(filters*4)
        self.conv8 = DoubleConv(filters*4, filters*2)
        
        self.upv9 = Up(filters*2, filters*1, mode='biliner')
        self.concat9 = Concat()
        self.ca9 = ChannelAttention(filters*2)
        self.conv9 = DoubleConv(filters*2, filters)
        
        self.conv10 = nn.Conv2d(filters, 3, kernel_size=1)
        # self.out = nn.Sigmoid()
        # Deep Supervision
        self.out8 = nn.Conv2d(filters*8, 3, kernel_size=1)
        self.out4 = nn.Conv2d(filters*4, 3, kernel_size=1)
        self.out2 = nn.Conv2d(filters*2, 3, kernel_size=1)
    
    def forward(self, R, I, mode='test'):
        I_att = 1-I
        x = torch.cat([R, I_att, I], dim=1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool1(conv1))        
        conv3 = self.conv3(self.pool2(conv2))
        conv4 = self.conv4(self.pool3(conv3))
        conv5 = self.conv5(self.pool4(conv4))
        
        d = self.dropout(conv5)

        up6 = self.upv6(d)
        if self.use_MaskMul:
            conv4 = MaskMul(8)(conv4, I_att)
        up6 = self.concat6(conv4, up6)
        up6 = self.ca6(up6) * up6
        conv6 = self.conv6(up6)

        up7 = self.upv7(conv6)
        if self.use_MaskMul:
            conv3 = MaskMul(4)(conv3, I_att)
        up7 = self.concat7(conv3, up7)
        up7 = self.ca7(up7) * up7
        conv7 = self.conv7(up7)

        up8 = self.upv8(conv7)
        if self.use_MaskMul:
            conv2 = MaskMul(2)(conv2, I_att)
        up8 = self.concat8(conv2, up8)
        up8 = self.ca8(up8) * up8
        conv8 = self.conv8(up8)

        up9 = self.upv9(conv8)
        if self.use_MaskMul:
            conv1 = MaskMul(1)(conv1, I_att)
        up9 = self.concat9(conv1, up9)
        up9 = self.ca9(up9) * up9
        conv9 = self.conv9(up9)
        
        out = self.conv10(conv9)
        
        if mode == 'train':
            out8 = self.out8(conv6)
            out4 = self.out4(conv7)
            out2 = self.out2(conv8)
            return out, out2, out4, out8
        else:
            return out


class Restore_D_Single(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        if activation == 'relu':
            self.relu = nn.ReLU()
        else:
            self.relu = nn.LeakyReLU(0.2)
        self.conv1 = DoubleConv(3, filters, stride=2)
        self.conv2 = DoubleConv(filters*1, filters*2, stride=2)
        self.conv3 = DoubleConv(filters*2, filters*4, stride=2)
        self.conv4 = DoubleConv(filters*4, filters*8, stride=2)
        self.conv5 = DoubleConv(filters*8, filters*16) # [b, 512, 1/16, 1/16]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dense = conv1x1(filters*16, filters*16)
        self.dropout = nn.Dropout(0.4)
        self.out = conv1x1(filters*16, 1)
    
    def forward(self, R):
        conv1 = self.conv1(R)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        dense = self.relu(self.dense(self.pool(conv5)))
        d = self.dropout(dense)
        out = self.out(d)
        return out


class Restore_D_Pyramid(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        if activation == 'relu':
            self.relu = nn.ReLU()
        else:
            self.relu = nn.LeakyReLU(0.2)
        self.conv1 = DoubleConv(3, filters, stride=2)
        self.R2_in = conv1x1(3, filters)
        self.conv2 = DoubleConv(filters*2, filters*2, stride=2)
        self.R4_in = conv1x1(3, filters)
        self.conv3 = DoubleConv(filters*3, filters*4, stride=2)
        self.R8_in = conv1x1(3, filters)
        self.conv4 = DoubleConv(filters*5, filters*8, stride=2)
        self.conv5 = DoubleConv(filters*8, filters*16) # [b, 512, 1/16, 1/16]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dense = conv1x1(filters*16, filters*16)
        self.dropout = nn.Dropout(0.4)
        self.out = conv1x1(filters*16, 1)
    
    def input_transfer(self, R_in, block, last_tensor):
        Rx_in = block(R_in)
        return torch.cat([Rx_in, last_tensor], dim=1)

    def forward(self, R, R2, R4, R8):
        conv1 = self.conv1(R)
        conv2_in = self.input_transfer(R2, self.R2_in, conv1)
        conv2 = self.conv2(conv2_in)
        conv3_in = self.input_transfer(R4, self.R4_in, conv2)
        conv3 = self.conv3(conv3_in)
        conv4_in = self.input_transfer(R8, self.R8_in, conv3)
        conv4 = self.conv4(conv4_in)
        conv5 = self.conv5(conv4)
        dense = self.relu(self.dense(self.pool(conv5)))
        d = self.dropout(dense)
        out = self.out(d)
        return out


class DenoiseNet(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.conv1 = DoubleConv(3, filters)
        self.pool1 = MaxPooling2D()
        
        self.conv2 = DoubleConv(filters, filters*2)
        self.pool2 = MaxPooling2D()
        
        self.conv3 = DoubleConv(filters*2, filters*4)
        self.pool3 = MaxPooling2D()
        
        self.conv4 = DoubleConv(filters*4, filters*8)
        self.pool4 = MaxPooling2D()
        
        self.conv5 = DoubleConv(filters*8, filters*16)
        self.dropout = nn.Dropout2d(0.5)

        self.upv6 = Up(filters*16, filters*8, mode='biliner')
        self.concat6 = Concat()
        # self.ca6 = ChannelAttention(filters*16)
        self.conv6 = DoubleConv(filters*16, filters*8)
        
        self.upv7 = Up(filters*8, filters*4, mode='biliner')
        self.concat7 = Concat()
        # self.ca7 = ChannelAttention(filters*8)
        self.conv7 = DoubleConv(filters*8, filters*4)
        
        self.upv8 = Up(filters*4, filters*2, mode='biliner')
        self.concat8 = Concat()
        # self.ca8 = ChannelAttention(filters*4)
        self.conv8 = DoubleConv(filters*4, filters*2)
        
        self.upv9 = Up(filters*2, filters*1, mode='biliner')
        self.concat9 = Concat()
        # self.ca9 = ChannelAttention(filters*2)
        self.conv9 = DoubleConv(filters*2, filters)
        
        self.conv10 = nn.Conv2d(filters, 3, kernel_size=1)
        # self.out = nn.Sigmoid()
        # Deep Supervision
        # self.out8 = nn.Conv2d(filters*8, 3, kernel_size=1)
        # self.out4 = nn.Conv2d(filters*4, 3, kernel_size=1)
        # self.out2 = nn.Conv2d(filters*2, 3, kernel_size=1)
    
    def forward(self, x, mode='test'):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool1(conv1))        
        conv3 = self.conv3(self.pool2(conv2))
        conv4 = self.conv4(self.pool3(conv3))
        conv5 = self.conv5(self.pool4(conv4))
        
        d = self.dropout(conv5)

        up6 = self.upv6(d)
        up6 = self.concat6(conv4, up6)
        # up6 = self.ca6(up6) * up6
        conv6 = self.conv6(up6)

        up7 = self.upv7(conv6)
        up7 = self.concat7(conv3, up7)
        # up7 = self.ca7(up7) * up7
        conv7 = self.conv7(up7)

        up8 = self.upv8(conv7)
        up8 = self.concat8(conv2, up8)
        # up8 = self.ca8(up8) * up8
        conv8 = self.conv8(up8)

        up9 = self.upv9(conv8)
        up9 = self.concat9(conv1, up9)
        # up9 = self.ca9(up9) * up9
        conv9 = self.conv9(up9)
        
        out = self.conv10(conv9)
        
        # if mode == 'train':
        #     out8 = self.out8(conv6)
        #     out4 = self.out4(conv7)
        #     out2 = self.out2(conv8)
        #     return out, out2, out4, out8
        # else:
        return out


class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, dep=20, num_filters=64, slope=0.2):
        '''
        Reference:
        K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang, "Beyond a Gaussian Denoiser: Residual
        Learning of Deep CNN for Image Denoising," TIP, 2017.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            dep (int): depth of the network, Default 20
            num_filters (int): number of filters in each layer, Default 64
        '''
        super().__init__()
        self.conv1 = conv3x3(in_channels, num_filters, bias=True)
        self.relu = nn.LeakyReLU(slope, inplace=True)
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3(num_filters, num_filters, bias=True))
            mid_layer.append(nn.LeakyReLU(slope, inplace=True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = conv3x3(num_filters, out_channels, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mid_layer(x)
        out = self.conv_last(x)

        return out


class UNet_VDN(nn.Module):
    def __init__(self, in_channels=3, out_channels=6, depth=4, wf=64, slope=0.2):
        """
        Reference:
        Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical
        Image Segmentation. MICCAI 2015.
        ArXiv Version: https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels, Default 3
            depth (int): depth of the network, Default 4
            wf (int): number of filters in the first layer, Default 32
        """
        super().__init__()
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, slope))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, slope))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, out_channels, bias=True)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


class KinD_noDecom(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        # self.decom_net = DecomNet()
        self.restore_net = RestoreNet_Unet()
        self.illum_net = IllumNet_Custom()
    
    def forward(self, R, I, ratio):
        I_final, I_standard = self.illum_net(I, ratio)
        R_final = self.restore_net(R, I)
        I_final_3 = torch.cat([I_final, I_final, I_final], dim=1)
        output = I_final_3 * R_final
        return R_final, I_final, output


class KinD(nn.Module):
    def __init__(self, filters=32, activation='lrelu', use_MaskMul=False):
        super().__init__()
        self.decom_net = DecomNet()
        self.restore_net = RestoreNet_Unet(use_MaskMul=use_MaskMul)
        self.illum_net = IllumNet_Custom()
    
    def forward(self, L, ratio, limit_highlight=True):
        R, I = self.decom_net(L)
        I_final, I_standard = self.illum_net(I, ratio)
        # I_final = I_standard
        R_final = self.restore_net(R, I)
        if limit_highlight:
            I_att = torch.clamp(F.sigmoid((.5-I)*10)/0.99330, min=0.2, max=1.0)
            R_final = R + (R_final-R) * torch.cat([I_att, I_att, I_att], dim=1)
        # I_final = I + (I_final-I) * (1-I/2)
        I_final_3 = torch.cat([I_final, I_final, I_final], dim=1)
        output = I_final_3 * R_final
        return R_final, I_final, output