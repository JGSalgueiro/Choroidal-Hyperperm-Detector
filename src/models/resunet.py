
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


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNet, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = DoubleConv(in_channels, 64)
        self.down_conv2 = DoubleConv(64, 128)
        self.down_conv3 = DoubleConv(128, 256)
        self.down_conv4 = DoubleConv(256, 512)
        self.up_transpose1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)  # Concatenating input from skip connection
        self.up_transpose2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)  # Concatenating input from skip connection
        self.up_transpose3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)  # Concatenating input from skip connection
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.residual_conv1 = nn.Conv2d(64, 64, kernel_size=1)  # Residual connection
        self.residual_conv2 = nn.Conv2d(128, 128, kernel_size=1)  # Residual connection
        self.residual_conv3 = nn.Conv2d(256, 256, kernel_size=1)  # Residual connection

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
        x = torch.cat([x, self.residual_conv3(x5)], dim=1)  # Adding residual connection
        x = self.up_conv1(x)
        x = self.up_transpose2(x)
        x = torch.cat([x, self.residual_conv2(x3)], dim=1)  # Adding residual connection
        x = self.up_conv2(x)
        x = self.up_transpose3(x)
        x = torch.cat([x, self.residual_conv1(x1)], dim=1)  # Adding residual connection
        x = self.up_conv3(x)
        x = self.out_conv(x)
        return x
