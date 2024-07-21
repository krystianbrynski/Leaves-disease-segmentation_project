import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = Conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        convolution = self.conv(x)
        maxpool = self.pool(convolution)
        return convolution, maxpool


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = Conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        level = torch.cat([x1, x2], dim=1)
        return self.conv(level)


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.bottom = Conv(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        conv1, pool1 = self.down1(x)
        conv2, pool2 = self.down2(pool1)
        conv3, pool3 = self.down3(pool2)
        conv4, pool4 = self.down4(pool3)

        bot = self.bottom(pool4)

        up1 = self.up1(bot, conv4)
        up2 = self.up2(up1, conv3)
        up3 = self.up3(up2, conv2)
        up4 = self.up4(up3, conv1)

        result = self.final(up4)

        return result
