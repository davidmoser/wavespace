"""
Input  shape : (B, 1, F, T)  – mono mag‑spectrogram
Output shape : (B, 1, F, T)
Final loss, 30 epochs, medley sample: 2.5
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
        self.enc1 = down(1, base_ch)  # /2   →  64 ch
        self.enc2 = down(base_ch, 2 * base_ch)  # /4   → 128 ch
        self.enc3 = down(2 * base_ch, 4 * base_ch)  # /8   → 256 ch

        self.bottleneck = nn.Sequential(  # /8
            nn.Conv2d(4 * base_ch, 8 * base_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(8 * base_ch, 8 * base_ch, 3, padding=1),
            nn.GELU()
        )  # 512 ch

        self.up3 = up(8 * base_ch, 2 * base_ch)  # 512→128, /8→/4
        self.dec3 = nn.Conv2d(4 * base_ch, 2 * base_ch, 3, padding=1)  # (128+128)=256 in

        self.up2 = up(2 * base_ch, base_ch)  # 128→64, /4→/2
        self.dec2 = nn.Conv2d(2 * base_ch, base_ch, 3, padding=1)  # (64+64)=128 in

        self.up1 = up(base_ch, base_ch // 2)  # 64→32, /2→/1
        self.dec1 = nn.Conv2d(base_ch // 2, 1, 1)  # 32 in → 1 out

    def forward(self, x):
        F, T = x.shape[-2:]
        if (F % 8) or (T % 8):
            raise RuntimeError(
                f"AudioUNet requires F and T divisible by 8 "
                f"(got F={F}, T={T}). Pad or crop your spectrogram."
            )

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(torch.cat([self.up3(b), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))
        d1 = self.dec1(self.up1(d2))
        return d1
