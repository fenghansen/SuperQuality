import torch
import torch.nn as nn
import torch.nn.functional as F
from base_layers import *


class Unet(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.conv1_1 = Conv2D(3, filters)
        self.conv1_2 = Conv2D(filters, filters)
        self.pool1 = MaxPooling2D()
        
        self.conv2_1 = Conv2D(filters, filters*2)
        self.conv2_2 = Conv2D(filters*2, filters*2)
        self.pool2 = MaxPooling2D()
        
        self.conv3_1 = Conv2D(filters*2, filters*4)
        self.conv3_2 = Conv2D(filters*4, filters*4)
        self.pool3 = MaxPooling2D()
        
        self.conv4_1 = Conv2D(filters*4, filters*8)
        self.conv4_2 = Conv2D(filters*8, filters*8)
        self.pool4 = MaxPooling2D()
        
        self.conv5_1 = Conv2D(filters*8, filters*16)
        self.conv5_2 = Conv2D(filters*16, filters*16)
        self.dropout = nn.Dropout2d(0.5)
        
        self.upv6 = ConvTranspose2D(filters*16, filters*8)
        self.concat6 = Concat()
        self.conv6_1 = Conv2D(filters*16, filters*8)
        self.conv6_2 = Conv2D(filters*8, filters*8)
        
        self.upv7 = ConvTranspose2D(filters*8, filters*4)
        self.concat7 = Concat()
        self.conv7_1 = Conv2D(filters*8, filters*4)
        self.conv7_2 = Conv2D(filters*4, filters*4)
        
        self.upv8 = ConvTranspose2D(filters*4, filters*2)
        self.concat8 = Concat()
        self.conv8_1 = Conv2D(filters*4, filters*2)
        self.conv8_2 = Conv2D(filters*2, filters*2)
        
        self.upv9 = ConvTranspose2D(filters*2, filters)
        self.concat9 = Concat()
        self.conv9_1 = Conv2D(filters*2, filters)
        self.conv9_2 = Conv2D(filters, filters)
        
        self.conv10_1 = nn.Conv2d(filters, 3, kernel_size=1, stride=1)
        self.out = nn.Sigmoid()
    
    def forward(self, x):
        # x = torch.cat([R, I], dim=1)
        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.pool1(conv2)
        
        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = self.pool1(conv3)
        
        conv4 = self.conv4_1(pool3)
        conv4 = self.conv4_2(conv4)
        pool4 = self.pool1(conv4)
        
        conv5 = self.conv5_1(pool4)
        conv5 = self.conv5_2(conv5)
        
        # d = self.dropout(conv5)
        up6 = self.upv6(conv5)
        up6 = self.concat6(conv4, up6)
        conv6 = self.conv6_1(up6)
        conv6 = self.conv6_2(conv6)
        
        up7 = self.upv7(conv6)
        up7 = self.concat7(conv3, up7)
        conv7 = self.conv7_1(up7)
        conv7 = self.conv7_2(conv7)
        
        up8 = self.upv8(conv7)
        up8 = self.concat8(conv2, up8)
        conv8 = self.conv8_1(up8)
        conv8 = self.conv8_2(conv8)
        
        up9 = self.upv9(conv8)
        up9 = self.concat9(conv1, up9)
        conv9 = self.conv9_1(up9)
        conv9 = self.conv9_2(conv9)
        
        conv10 = self.conv10_1(conv9)
        out = self.out(conv10)
        return out


