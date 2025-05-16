"""
Input  shape : (B, 1, F, T)  – mono mag‑spectrogram
Output shape : (B, 1, F, T)
"""
import torch
import torch.nn as nn

def down(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
        nn.MaxPool2d(2)
    )

def up(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
        nn.GELU()
    )

class AudioUNet(nn.Module):
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.enc1 = down(1,  base_ch)
        self.enc2 = down(base_ch, 2*base_ch)
        self.enc3 = down(2*base_ch, 4*base_ch)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(4*base_ch, 8*base_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(8*base_ch, 4*base_ch, 3, padding=1),
            nn.GELU()
        )
        self.up3  = up(8*base_ch, 2*base_ch)
        self.dec3 = nn.Conv2d(4*base_ch, 2*base_ch, 3, padding=1)
        self.up2  = up(2*base_ch, base_ch)
        self.dec2 = nn.Conv2d(2*base_ch, base_ch, 3, padding=1)
        self.up1  = up(base_ch, base_ch//2)
        self.dec1 = nn.Conv2d(base_ch, 1, 1)          # linear output

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(e3)
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return d1
