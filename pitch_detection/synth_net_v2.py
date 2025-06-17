import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pitch_detection.configuration import Configuration


class SynthNet(nn.Module):
    """Depth-wise 2-D conv along frequency and time."""

    def __init__(self, cfg: Configuration) -> None:
        super().__init__()
        self.channels = cfg.out_ch
        self.kernel_f_len = cfg.kernel_f_len
        self.kernel_t_len = cfg.kernel_t_len

        inv_kernel_value = np.log(np.exp(cfg.kernel_value) - 1)
        if cfg.kernel_random:
            self.kernel = nn.Parameter(
                inv_kernel_value
                - torch.rand(self.channels, 1, self.kernel_f_len, self.kernel_t_len),
            )
        else:
            self.kernel = nn.Parameter(
                torch.ones(self.channels, 1, self.kernel_f_len, self.kernel_t_len)
                * inv_kernel_value,
            )
        if cfg.init_f0 == "point":
            inv_one = np.log(np.exp(1) - 1)
            if cfg.force_f0:
                raise ValueError("Only init_f0 or force_f0 can be True, but not both")
            with torch.no_grad():
                self.kernel[:, :, 0, 0] = inv_one
        elif cfg.init_f0 == "exponential":
            if cfg.force_f0:
                raise ValueError("Only init_f0 or force_f0 can be True, but not both")
            with torch.no_grad():
                C = self.channels
                T = self.kernel_t_len
                if C == 1 or T == 1:
                    weights = torch.ones(C, T)
                else:
                    t = torch.arange(T, dtype=torch.float32).view(1, T)
                    c = torch.arange(C, dtype=torch.float32).view(C, 1)
                    alpha = np.log(1000.0) / ((C - 1) * (T - 1))
                    weights = torch.exp(-alpha * c * t)
                    weights = weights / weights.sum(dim=1, keepdim=True)
                inv = torch.log(torch.exp(weights) - 1)
                self.kernel[:, :, 0, :] = inv
        elif cfg.init_f0 not in ("none", "point", "exponential"):
            raise ValueError(f"Unknown init_f0 option: {cfg.init_f0}")
        self.force_f0 = cfg.force_f0

    def forward(self, x):  # (B,C,F,T)
        B, C, F_, T = x.shape
        eff_kernel_f_len = self.kernel_f_len + (1 if self.force_f0 else 0)
        x = F.pad(x, (self.kernel_t_len - 1, 0, eff_kernel_f_len - 1, 0))

        weight = F.softplus(self.kernel)
        if self.force_f0:
            ones = torch.ones(C, 1, 1, self.kernel_t_len, device=x.device)
            weight = torch.cat([ones, weight], dim=2)

        weight = torch.flip(weight, dims=[-1, -2])
        y = F.conv2d(x, weight, groups=C)
        y = y.mean(dim=1, keepdim=True)
        return y  # (B,1,F,T)
