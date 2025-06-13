import re
import torch
from torch import nn

from pitch_detection.configuration import Configuration
from pitch_detection.pitch_det_net_v1 import PitchDetNet as PitchDetNetV1
from pitch_detection.pitch_det_net_v2 import PitchDetNet as PitchDetNetV2
from pitch_detection.synth_net_v1 import SynthNet as SynthNetV1
from pitch_detection.synth_net_v2 import SynthNet as SynthNetV2

_PITCH_DET_REGISTRY = {1: PitchDetNetV1, 2: PitchDetNetV2}
_SYNTH_REGISTRY = {1: SynthNetV1, 2: SynthNetV2}


def get_pitch_det_model(version: str, cfg: Configuration) -> nn.Module:
    v_num = int(re.search(r"v(\d+)_?", version).group(1))
    try:
        net_cls = _PITCH_DET_REGISTRY[v_num]
    except KeyError as e:
        raise ValueError(f"Unknown pitch_det_net version {version}") from e
    return net_cls(base_ch=cfg.base_ch, out_ch=cfg.out_ch)


def get_synth_model(version: str, cfg: Configuration) -> nn.Module:
    v_num = int(re.search(r"v(\d+)_?", version).group(1))
    try:
        net_cls = _SYNTH_REGISTRY[v_num]
    except KeyError as e:
        raise ValueError(f"Unknown synth_net version {version}") from e
    return net_cls(cfg)


class PitchAutoencoder(nn.Module):
    """U-Net → soft-plus → Synthesizer → channel-sum."""

    def __init__(self, cfg: Configuration):
        super().__init__()
        self.pitch_det_net = get_pitch_det_model(cfg.pitch_det_version, cfg)
        self.synth = get_synth_model(cfg.synth_net_version, cfg)

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
