import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SynthNet(nn.Module):
    """
    Depth-wise 1-D conv along frequency.
    Kernel length L should cover the maximum overtone distance in bins.
    Fundamental weight is frozen to 1; overtones are constrained to (0,1).
    """

    def __init__(self, channels: int = 32, kernel_len: int = 128, force_f0: bool = False, kernel_random: bool = False,
                 kernel_value: float = 0.1) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_len = kernel_len
        # positive convolution kernels as raw parameters

        inv_kernel_value = np.log(np.exp(kernel_value) - 1)
        if kernel_random:
            self.kernel = nn.Parameter(inv_kernel_value - torch.rand(channels, 1, kernel_len))
        else:
            self.kernel = nn.Parameter(torch.ones(channels, 1, kernel_len) * inv_kernel_value)
        self.force_f0 = force_f0

    def forward(self, x):  # (B,32,F,T)
        B, C, F_, T = x.shape
        # over = torch.sigmoid(self.theta)  # (C,L-1) in (0,1) # if we pin f0 to 1
        # kernel = torch.cat([torch.ones(C, 1, device=x.device), over], dim=1) # if we pin f0 to 1
        # kernel = kernel.view(C, 1, -1)  # (C,1,L) # if we pin f0 to 1
        # kernel = self.theta

        # (B, C, F, T) -> (B, T, C, F) -> (B*T, C, F)
        x_ft = x.permute(0, 3, 1, 2).contiguous().view(B * T, C, F_)
        eff_kernel_len = self.kernel_len + (1 if self.force_f0 else 0)
        x_ft = F.pad(x_ft, (eff_kernel_len - 1, 0))

        # use positive kernels via softplus transformation
        weight = F.softplus(self.kernel)
        if self.force_f0:
            weight = torch.concat([torch.ones(C, 1, 1, device=x.device), weight], dim=2)

        # flip so that index 0 corresponds to lowest frequency
        weight = torch.flip(weight, dims=[-1])
        y = F.conv1d(x_ft, weight, groups=C)
        y = y.view(B, T, C, F_).permute(0, 2, 3, 1).contiguous()
        y = y.mean(dim=1, keepdim=True)  # sum channels
        return y  # (B,1,F,T)
