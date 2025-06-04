import torch.nn.functional as F
from torch import nn

from pitch_detection.pitch_det_net import PitchDetNet
from pitch_detection.synth_net import SynthNet


class PitchAutoencoder(nn.Module):
    """U-Net → soft-plus → Synthesizer → channel-sum."""

    def __init__(self, base_ch=16, out_ch=32, kernel_len=128):
        super().__init__()
        self.f0_net = PitchDetNet(base_ch, out_ch)
        self.synth = SynthNet(out_ch, kernel_len)

    def forward(self, x):
        f = F.softplus(self.f0_net(x))  # ensure a ≥ 0
        s = self.synth(f)  # (B,1,F,T)
        return s, f


def entropy_term(a, eps=1e-12):
    """Mean Shannon entropy over (channel,time)."""
    p = a / (a.sum(dim=2, keepdim=True) + eps)
    h = -(p * (p + eps).log()).sum(dim=2)  # (B,C,T)
    return h.mean()


def laplacian_1d(a):
    """|∇²_f a| for sharpness (B,C,F,T) → scalar."""
    return (a[:, :, 2:] - 2 * a[:, :, 1:-1] + a[:, :, :-2]).abs().mean()
