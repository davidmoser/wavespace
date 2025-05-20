"""
Input  shape : (B, 1, F, T)  – mono mag‑spectrogram
Output shape : (B, 1, F, T)
Final loss, 30 epochs, medley sample:
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
        self.enc4 = down(4 * base_ch, 8 * base_ch)  # /16 -> 512 ch
        self.up4 = up(8 * base_ch, 4 * base_ch)  # /8 -> 256
        self.up3 = up(8 * base_ch, 2 * base_ch)  # 512→128, /8→/4
        self.up2 = up(4 * base_ch, base_ch)  # 256→64, /4→/2
        self.up1 = up(2 * base_ch, 1)  # 128→1, /2→/1

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
        e4 = self.enc4(e3)
        d4 = self.up4(e4)
        d3 = self.up3(torch.cat([d4, e3], 1))
        d2 = self.up2(torch.cat([d3, e2], 1))
        d1 = self.up1(torch.cat([d2, e1], 1))
        return d1
