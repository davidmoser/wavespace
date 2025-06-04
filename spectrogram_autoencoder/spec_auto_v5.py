import torch
import torch.nn as nn


def down(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
        nn.BatchNorm1d(out_ch),
        nn.GELU(),
        nn.MaxPool1d(4)
    )


def up(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose1d(in_ch, in_ch, kernel_size=4, stride=4),
        nn.GELU(),
        nn.Conv1d(in_ch, out_ch, 5, padding=2)
    )


class SpecAutoNet(nn.Module):
    def __init__(self, base_ch: int = 16, out_ch: int = 1):
        super().__init__()
        # Initially: F=256, C=1
        self.enc1 = down(1, base_ch)  # 64, b
        self.enc2 = down(base_ch, 4 * base_ch)  # 16, 4b
        self.enc3 = down(4 * base_ch, 16 * base_ch)  # 4, 16b
        self.enc4 = down(16 * base_ch, 64 * base_ch)  # 1, 64b

        self.up4 = up(64 * base_ch, 16 * base_ch)  # 4, 16b
        self.up3 = up(16 * base_ch, 4 * base_ch)  # 16, 4b
        self.up2 = up(4 * base_ch, base_ch)  # 64, b
        self.up1 = up(base_ch, base_ch)  # 256, out

        self.dec4 = nn.Conv1d(32 * base_ch, 16 * base_ch, 5, padding=2)
        self.dec3 = nn.Conv1d(8 * base_ch, 4 * base_ch, 5, padding=2)
        self.dec2 = nn.Conv1d(2 * base_ch, base_ch, 5, padding=2)
        self.dec1 = nn.Conv1d(base_ch, out_ch, 1)
        self.out_ch = out_ch

    def forward(self, x, return_latent: bool = False):
        B, C, F, T = x.shape
        if F % 256:
            raise RuntimeError(f"F must be divisible by 256 (got {F}).")

        # work on (B·T, C, F)
        x = x.permute(0, 3, 1, 2).contiguous().view(B * T, C, F)

        e1 = self.enc1(x)  # b
        e2 = self.enc2(e1)  # 4b
        e3 = self.enc3(e2)  # 16 b
        e4 = self.enc4(e3)  # 64b

        d4 = self.up4(e4)  # 16b
        d3 = self.up3(self.dec4(torch.cat([d4, e3], 1)))  # 16b + 16b -> 16b -> 4b
        d2 = self.up2(self.dec3(torch.cat([d3, e2], 1)))  # 4b + 4b -> 4b -> b
        d1 = self.up1(self.dec2(torch.cat([d2, e1], 1)))  # b + b -> b -> b
        d0 = self.dec1(d1)  # b -> out

        # restore shape (B, C=1, F, T)
        d0 = d0.view(B, T, self.out_ch, F).permute(0, 2, 3, 1).contiguous()
        if return_latent:
            lat = e4.view(B, T, -1, F // 16).permute(0, 2, 3, 1).contiguous()
            return d0, lat
        return d0
