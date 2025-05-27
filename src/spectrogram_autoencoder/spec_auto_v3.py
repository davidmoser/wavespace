"""
2D Convnet, residual connections
Final loss, 10 epochs, medley sample: 1.8
"""
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


class SpecAutoNet(nn.Module):
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.enc1 = down(1, base_ch)  # /2   →  64 ch
        self.enc2 = down(base_ch, 2 * base_ch)  # /4   → 128 ch
        self.enc3 = down(2 * base_ch, 4 * base_ch)  # /8   → 256 ch
        self.enc4 = down(4 * base_ch, 8 * base_ch)  # /16 -> 512 ch
        self.up4 = up(8 * base_ch, 4 * base_ch)  # /8 -> 256
        self.up3 = up(4 * base_ch, 2 * base_ch)  # 512→128, /8→/4
        self.up2 = up(2 * base_ch, base_ch)  # 256→64, /4→/2
        self.up1 = up(base_ch, base_ch // 2)  # 128→1, /2→/1
        self.dec4 = nn.Conv2d(4 * base_ch, 4 * base_ch, 3, padding=1)
        self.dec3 = nn.Conv2d(2 * base_ch, 2 * base_ch, 3, padding=1)
        self.dec2 = nn.Conv2d(1 * base_ch, 1 * base_ch, 3, padding=1)
        self.dec1 = nn.Conv2d(base_ch // 2, 1, 1)

    def forward(self, x, return_latent: bool = False):
        F, T = x.shape[-2:]
        if (F % 16) or (T % 16):
            raise RuntimeError(
                f"SpecAutoNet requires F and T divisible by 8 "
                f"(got F={F}, T={T}). Pad or crop your spectrogram."
            )

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d4 = self.up4(e4)
        d3 = self.up3(self.dec4(d4))
        d2 = self.up2(self.dec3(d3))
        d1 = self.up1(self.dec2(d2))
        d0 = self.dec1(d1)
        return (d0, e4) if return_latent else d0
