import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os
###########################################################################
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        # two 3X3 equal one 5X5
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)
#########################################################################################
class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)
#########################################################################################
class ResNest(torch.nn.Module):
    def __init__(self, in_ch, channel_G):
        super(ResNest, self).__init__()
        self.head = SingleConv(in_ch, channel_G)
        self.conv_left1 = SingleConv(channel_G, channel_G)
        self.conv_left2 = SingleConv(channel_G, channel_G)
        self.conv_left3 = SingleConv(channel_G, channel_G)
        self.conv_left4 = SingleConv(channel_G, channel_G)
        self.conv_remainder = SingleConv(channel_G, channel_G)
        self.conv_right4 = SingleConv(channel_G, channel_G)
        self.conv_right3 = SingleConv(channel_G, channel_G)
        self.conv_right2 = SingleConv(channel_G, channel_G)
        self.conv_right1 = SingleConv(channel_G, channel_G)
        self.tail = SingleConv(channel_G, 1)

    def forward(self, x):
        x_head = self.head(x)
        c1 = self.conv_left1(x_head)
        c2 = self.conv_left2(c1)
        c3 = self.conv_left3(c2)
        c4 = self.conv_left4(c3)
        remainder = self.conv_remainder(c4)
        c_r4 = self.conv_right4(remainder + c4)
        c_r3 = self.conv_right3(c_r4 + c3)
        c_r2 = self.conv_right2(c_r3 + c2)
        c_r1 = self.conv_right1(c_r2 + c1)
        c_r1_2 = self.tail(c_r1 + x_head)
        return c_r1_2  # residual
###########################################################################
class ResUnet(nn.Module):
    def __init__(self, in_ch, channel_G):
        super(ResUnet, self).__init__()
        self.head = SingleConv(in_ch, channel_G)
        self.conv_left1 = DoubleConv(channel_G, channel_G)
        self.pool1 = nn.Conv2d(channel_G, channel_G, kernel_size=2, stride=2, padding=0)  # down sampling
        self.conv_left2 = DoubleConv(channel_G, channel_G)
        self.pool2 = nn.Conv2d(channel_G, channel_G, kernel_size=2, stride=2, padding=0)
        self.conv_left3 = DoubleConv(channel_G, channel_G)
        self.pool3 = nn.Conv2d(channel_G, channel_G, kernel_size=2, stride=2, padding=0)
        self.conv_left4 = DoubleConv(channel_G, channel_G)
        self.pool4 = nn.Conv2d(channel_G, channel_G, kernel_size=2, stride=2, padding=0)
        self.conv_remainder = DoubleConv(channel_G, channel_G)
        self.up4 = nn.ConvTranspose2d(channel_G, channel_G, 2, stride=2)
        self.conv_right4 = DoubleConv(2 * channel_G, channel_G)
        self.up3 = nn.ConvTranspose2d(channel_G, channel_G, 2, stride=2)
        self.conv_right3 = DoubleConv(2*channel_G, channel_G)
        self.up2 = nn.ConvTranspose2d(channel_G, channel_G, 2, stride=2)
        self.conv_right2 = DoubleConv(2*channel_G, channel_G)
        self.up1 = nn.ConvTranspose2d(channel_G, channel_G, 2, stride=2)
        self.conv_right1 = DoubleConv(2*channel_G, channel_G)
        self.tail = SingleConv(channel_G, 1)

    def forward(self, x):
        x_head = self.head(x)
        c1 = self.conv_left1(x_head)
        p1 = self.pool1(c1)
        c2 = self.conv_left2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv_left3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv_left4(p3)
        p4 = self.pool4(c4)
        remainder = self.conv_remainder(p4)
        up_4 = self.up4(remainder + p4) # residual
        merge4 = torch.cat([up_4, c4], dim=1)
        c_r4 = self.conv_right4(merge4)
        up_3 = self.up3(c_r4 + p3) # residual
        merge3 = torch.cat([up_3, c3], dim=1)
        c_r3 = self.conv_right3(merge3)
        up_2 = self.up2(c_r3 + p2) # residual
        merge2 = torch.cat([up_2, c2], dim=1)
        c_r2 = self.conv_right2(merge2)
        up_1 = self.up1(c_r2 + p1)  # residual
        merge1 = torch.cat([up_1, c1], dim=1)
        c_r1 = self.conv_right1(merge1)
        c_r1_2 = self.tail(c_r1 + x_head)
        return c_r1_2     # residual
###########################################################################
class DualNet(nn.Module):
    def __init__(self, growth_rate):
        super(DualNet, self).__init__()
        self.G = growth_rate
        self.rdb = ResNest(3, self.G)  # local residual learning

    def forward(self, z, Ax, z0):
        x = torch.cat([z, Ax, z0], dim=1)
        x = self.rdb(x)
        return x
###########################################################################
class DualNet_h(nn.Module):
    def __init__(self, growth_rate):
        super(DualNet_h, self).__init__()
        self.G = growth_rate
        self.rdb = ResNest(3, self.G)  # local residual learning

    def forward(self, zh, Ahx, zh0):
        x = torch.cat([zh, Ahx, zh0], dim=1)
        x = self.rdb(x)
        return x
###########################################################################
class PrimalNet(nn.Module):
    def __init__(self,growth_rate):
        super(PrimalNet, self).__init__()
        self.G = growth_rate
        self.rdb = ResUnet(3, self.G)  # local residual learning

    def forward(self, x, ATz, AhTz):
        x = torch.cat([x, ATz, AhTz], dim=1)
        x = self.rdb(x)
        return x
###########################################################################