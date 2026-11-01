import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.layers(x)

class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.bottleneck = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(128 + 64, 64)
        self.dec2 = ConvBlock(64 + 32, 32)
        self.dec1 = ConvBlock(32 + 16, 16)
        self.head = nn.Conv2d(16, 4, 1)
    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        x = self.bottleneck(self.pool(c))
        for skip, decoder in [(c, self.dec3), (b, self.dec2), (a, self.dec1)]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)

