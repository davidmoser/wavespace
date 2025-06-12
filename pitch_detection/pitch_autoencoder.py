import torch
from torch import nn

from pitch_detection.configuration import Configuration
from pitch_detection.pitch_det_net_v2 import PitchDetNet
from pitch_detection.synth_net import SynthNet


class PitchAutoencoder(nn.Module):
    """U-Net → soft-plus → Synthesizer → channel-sum."""

    def __init__(self, cfg: Configuration):
        super().__init__()
        self.pitch_det_net = PitchDetNet(base_ch=cfg.base_ch, out_ch=cfg.out_ch)
        self.synth = SynthNet(channels=cfg.out_ch, kernel_len=cfg.kernel_len, force_f0=cfg.force_f0,
                              kernel_random=cfg.kernel_random)

    def forward(self, x):
        # f = F.softplus(self.pitch_det_net(x))  # ensure a ≥ 0
        f = self.pitch_det_net(x)
        s = self.synth(f)  # (B,1,F,T)
        return s, f


def entropy_term(a, eps=1e-12):
    """Mean Shannon entropy over (channel,time)."""
    p = a / (a.sum(dim=2, keepdim=True) + eps)
    h = -(p * (p + eps).log()).sum(dim=2)  # (B,C,T)
    return (h * a.sum(dim=2)).mean()


def laplacian_1d(a):
    """|∇²_f a| for sharpness (B,C,F,T) → scalar."""
    return (a[:, :, 2:] - 2 * a[:, :, 1:-1] + a[:, :, :-2]).abs().mean()


def distribution_std(a, eps: float = 1e-12):
    """Standard deviation of |a| considered as a probability distribution.

    The tensor is normalized along the frequency dimension so that values sum
    to 1 for each (batch, channel, time) pair. The returned scalar is the mean
    standard deviation over all distributions.
    """
    B, C, F, T = a.shape
    weights = a.abs()
    weights = weights / (weights.sum(dim=2, keepdim=True) + eps)
    idx = torch.arange(F, device=a.device, dtype=a.dtype).view(1, 1, F, 1)
    mean = (weights * idx).sum(dim=2, keepdim=True)
    var = (weights * (idx - mean) ** 2).sum(dim=2)
    return var.sqrt().mean()
