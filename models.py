import torch
import torch.nn as nn
import torch.nn.functional as F
from base_layers import *

class H_restore(nn.Module):
    def __init__(self, filters=16, activation='relu', w_res=0.3):
        super().__init__()
        self.w = torch.tensor(w_res, requires_grad=False)


class Unet_mini(nn.Module):
    def __init__(self, filters=16, activation='relu', w_res=0.3):
        super().__init__()
        self.w = torch.tensor(w_res, requires_grad=False, device='cuda')
        self.conv1 = DoubleConv(2, filters, activation)
        self.pool1 = AvgPooling2D()
        
        self.conv2 = DoubleConv(filters, filters*2, activation)
        self.pool2 = AvgPooling2D()
        
        self.conv3 = DoubleConv(filters*2, filters*4, activation)
        self.pool3 = AvgPooling2D()
        
        self.conv4 = DoubleConv(filters*4, filters*8, activation)

        self.dropout = nn.Dropout2d(0.5)
        
        self.upv7 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.concat7 = Concat()
        self.conv7 = DoubleConv(filters*12, filters*4, activation)
        
        self.upv8 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.concat8 = Concat()
        self.conv8 = DoubleConv(filters*6, filters*2, activation)
        
        self.upv9 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.concat9 = Concat()
        self.conv9 = DoubleConv(filters*3, filters, activation)
        
        self.conv10 = nn.Conv2d(filters, 2, kernel_size=1, stride=1)
        self.out = nn.Sigmoid()

    def set_w(self, w=None):
        if w is None:
            self.w.requires_grad = True
        else:
            self.w = torch.tensor(w, requires_grad=False, device='cuda')
    
    def get_w(self):
        try:
            w = self.w.numpy()
        except TypeError:
            w = self.w.detach().cpu().numpy()
        return w
    
    def forward(self, x):
        x = x[:,:-1,:,:]
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool1(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool1(conv3)
        
        conv4 = self.conv4(pool3)
        
        d = self.dropout(conv4)
        
        up7 = self.upv7(d)
        up7 = self.concat7(conv3, up7)
        conv7 = self.conv7(up7)
        
        up8 = self.upv8(conv7)
        up8 = self.concat8(conv2, up8)
        conv8 = self.conv8(up8)
        
        up9 = self.upv9(conv8)
        up9 = self.concat9(conv1, up9)
        conv9 = self.conv9(up9)
        
        conv10 = self.conv10(conv9)
        res_out = self.out(conv10)
        out = self.w * x + res_out
        out.clamp_(min=0.0, max=1.0)
        return out


class Unet(nn.Module):
    def __init__(self, filters=32, activation='relu'):
        super().__init__()
        self.conv1 = DoubleConv(3, filters, activation)
        self.pool1 = MaxPooling2D()
        
        self.conv2 = DoubleConv(filters, filters*2, activation)
        self.pool2 = MaxPooling2D()
        
        self.conv3 = DoubleConv(filters*2, filters*4, activation)
        self.pool3 = MaxPooling2D()
        
        self.conv4 = DoubleConv(filters*4, filters*8, activation)
        self.pool4 = MaxPooling2D()
        
        self.conv5 = DoubleConv(filters*8, filters*16, activation)
        self.dropout = nn.Dropout2d(0.5)
        
        self.upv6 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.concat6 = Concat()
        self.conv6 = DoubleConv(filters*24, filters*8, activation)
        
        self.upv7 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.concat7 = Concat()
        self.conv7 = DoubleConv(filters*12, filters*4, activation)
        
        self.upv8 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.concat8 = Concat()
        self.conv8 = DoubleConv(filters*6, filters*2, activation)
        
        self.upv9 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.concat9 = Concat()
        self.conv9 = DoubleConv(filters*3, filters, activation)
        
        self.conv10 = nn.Conv2d(filters, 2, kernel_size=1, stride=1)
        self.out = nn.Sigmoid()
    
    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool1(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool1(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool1(conv4)
        
        conv5 = self.conv5(pool4)
        
        d = self.dropout(conv5)
        up6 = self.upv6(d)
        up6 = self.concat6(conv4, up6)
        conv6 = self.conv6(up6)
        
        up7 = self.upv7(conv6)
        up7 = self.concat7(conv3, up7)
        conv7 = self.conv7(up7)
        
        up8 = self.upv8(conv7)
        up8 = self.concat8(conv2, up8)
        conv8 = self.conv8(up8)
        
        up9 = self.upv9(conv8)
        up9 = self.concat9(conv1, up9)
        conv9 = self.conv9(up9)
        
        conv10 = self.conv10(conv9)
        out = self.out(conv10)
        return out


