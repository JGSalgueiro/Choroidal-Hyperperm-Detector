import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DRUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DRUNet, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = DoubleConv(in_channels, 64)
        self.down_conv2 = DoubleConv(64, 128)
        self.down_conv3 = DoubleConv(128, 256)
        self.down_conv4 = DoubleConv(256, 512)
        self.up_transpose1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256) 
        self.up_transpose2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)  
        self.up_transpose3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)  
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.se_block1 = SELayer(64) 
        self.se_block2 = SELayer(128) 
        self.se_block3 = SELayer(256) 
        self.se_block4 = SELayer(512) 

    def forward(self, x):
        # Encoder
        x1 = self.down_conv1(x)
        x2 = self.max_pool(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool(x5)
        x7 = self.down_conv4(x6)

        # Decoder
        x = self.up_transpose1(x7)
        x = torch.cat([x, self.se_block4(x5)], dim=1) 
        x = self.up_conv1(x)
        x = self.up_transpose2(x)
        x = torch.cat([x, self.se_block3(x3)], dim=1) 
        x = self.up_conv2(x)
        x = self.up_transpose3(x)
        x = torch.cat([x, self.se_block2(x1)], dim=1) 
        x = self.up_conv3(x)
        x = self.out_conv(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
