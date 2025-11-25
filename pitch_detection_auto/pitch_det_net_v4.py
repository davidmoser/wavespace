import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv → ReLU) ×2 using 2D convolutions"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """2×2 down-sampling along frequency and time, then double conv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """2×2 up-sampling along frequency and time, concat, double conv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x1, x2):
        # x1: deeper feature, x2: skip connection
        x1 = self.up(x1)

        diff_f = x2.size(-2) - x1.size(-2)
        diff_t = x2.size(-1) - x1.size(-1)

        if diff_f or diff_t:
            x1 = F.pad(
                x1,
                (
                    diff_t // 2,
                    diff_t - diff_t // 2,
                    diff_f // 2,
                    diff_f - diff_f // 2,
                ),
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PitchDetNet(nn.Module):
    """UNet-style network using 2D convolutions on both frequency and time.

    Input  : (B, C_in, F, T)
    Output : (B, C_out, F, T)
    """

    def __init__(self, in_ch=1, base_ch=4, out_ch=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        self.up1 = Up(base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch)

        self.outc = OutConv(base_ch, out_ch)

    def forward(self, x):
        # x: (B, C, F, T)
        _, _, F_, T_ = x.shape

        # We have 3 downsamplings with factor 2 → need multiples of 8 in both dims
        pad_f = (-F_) % 8
        pad_t = (-T_) % 8

        if pad_f or pad_t:
            # pad last two dims: (left_T, right_T, top_F, bottom_F)
            x = F.pad(x, (0, pad_t, 0, pad_f))
            F_out = F_ + pad_f
            T_out = T_ + pad_t
        else:
            F_out = F_
            T_out = T_

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.outc(x)

        # Crop back to original F,T
        if pad_f or pad_t:
            x = x[:, :, :F_, :T_]

        return x
