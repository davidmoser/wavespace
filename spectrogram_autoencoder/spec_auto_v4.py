"""
1D-Convnet, no residual connections
Final loss, 30 epochs, medley: 4.2
"""
import torch.nn as nn


def down1d(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2, stride=4),
        nn.BatchNorm1d(out_ch),
        nn.GELU(),
        # nn.MaxPool1d(4)
    )


def up1d(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose1d(in_ch, in_ch, kernel_size=4, stride=4),
        nn.GELU(),
        nn.Conv1d(in_ch, out_ch, 5, padding=2)
    )


class SpecAutoNet(nn.Module):
    def __init__(self, base_ch: int = 4):
        super().__init__()
        # Initially: F=256, C=1
        self.enc1 = down1d(1, base_ch)  # 64, 4
        self.enc2 = down1d(base_ch, 4 * base_ch)  # 16, 16
        self.enc3 = down1d(4 * base_ch, 16 * base_ch)  # 4, 64
        self.enc4 = down1d(16 * base_ch, 64 * base_ch)  # 1, 256

        self.dec4 = up1d(64 * base_ch, 16 * base_ch)  # 4, 64
        self.dec3 = up1d(16 * base_ch, 4 * base_ch)  # 16, 16
        self.dec2 = up1d(4 * base_ch, base_ch)  # 64, 4
        self.dec1 = up1d(base_ch, 1)  # 256, 1

    def forward(self, x, return_latent: bool = False):
        """
        x : (B, 1, F, T)  – mono magnitude spectrogram
        All ops act along F; T is handled by reshaping.
        """
        B, C, F, T = x.shape
        if F % 16:
            raise RuntimeError(f"F={F} must be divisible by 16.")

        # (B, 1, F, T) → (B·T, 1, F)
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, F)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)

        # back to (B, 1, F, T)
        d1 = d1.reshape(B, T, 1, F).permute(0, 2, 3, 1).contiguous()
        return (d1, e4) if return_latent else d1
