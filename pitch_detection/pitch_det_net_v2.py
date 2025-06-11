import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv → ReLU) ×2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """4× down-sampling then double conv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """4× up-sampling, concat, double conv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # take in_ch-channels, output out_ch-channels
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=4)
        # after concat we have out_ch (skip) + out_ch (upsampled) = 2·out_ch
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size(-1) - x1.size(-1)
        if diff:
            x1 = F.pad(x1, (diff // 2, diff - diff // 2))
        x = torch.cat([x2, x1], dim=1)  # channel dim
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PitchDetNet(nn.Module):
    """
    Input  : (B,C_in,F,T)
    Output : (B,C_out,F,T)
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
        self.out_act = nn.Softplus()

    def forward(self, x):
        B, C, F_, T = x.shape
        # reshape so time slices are batch instances
        x = x.permute(0, 3, 1, 2).contiguous().view(B * T, C, F_)

        # optional: pad F so it is divisible by 64
        pad = (-F_) % 64
        if pad:
            x = F.pad(x, (0, pad))
            F_out = F_ + pad
        else:
            F_out = F_

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        x = self.out_act(x)

        # remove padding and restore original shape
        if pad:
            x = x[..., :F_]
        x = x.view(B, T, -1, F_).permute(0, 2, 3, 1)  # (B,C_out,F,T)
        return x
