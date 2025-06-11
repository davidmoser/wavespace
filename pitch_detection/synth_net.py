import torch
import torch.nn.functional as F
from torch import nn


class SynthNet(nn.Module):
    """
    Depth-wise 1-D conv along frequency.
    Kernel length L should cover the maximum overtone distance in bins.
    Fundamental weight is frozen to 1; overtones are constrained to (0,1).
    """

    def __init__(self, channels: int = 32, kernel_len: int = 128):
        super().__init__()
        self.channels = channels
        self.kernel_len = kernel_len
        # positive convolution kernels as raw parameters
        self.kernel = nn.Parameter(torch.zeros(channels, 1, kernel_len))
        self.reset_parameters()

    def reset_parameters(self):
        inv_softplus_one = torch.log(torch.expm1(torch.tensor(1.0)))
        with torch.no_grad():
            self.kernel.fill_(-10.0)
            self.kernel[:, :, 0] = inv_softplus_one

    def forward(self, x):  # (B,32,F,T)
        B, C, F_, T = x.shape
        # over = torch.sigmoid(self.theta)  # (C,L-1) in (0,1) # if we pin f0 to 1
        # kernel = torch.cat([torch.ones(C, 1, device=x.device), over], dim=1) # if we pin f0 to 1
        # kernel = kernel.view(C, 1, -1)  # (C,1,L) # if we pin f0 to 1
        # kernel = self.theta

        # (B, C, F, T) -> (B, T, C, F) -> (B*T, C, F)
        x_ft = x.permute(0, 3, 1, 2).contiguous().view(B * T, C, F_)
        x_ft = F.pad(x_ft, (self.kernel_len - 1, 0))

        # use positive kernels via softplus transformation
        weight = F.softplus(self.kernel)
        # flip so that index 0 corresponds to lowest frequency
        weight = torch.flip(weight, dims=[-1])
        y = F.conv1d(x_ft, weight, groups=C)
        y = y.view(B, T, C, F_).permute(0, 2, 3, 1).contiguous()
        y = y.sum(dim=1, keepdim=True)  # sum channels
        return y  # (B,1,F,T)
