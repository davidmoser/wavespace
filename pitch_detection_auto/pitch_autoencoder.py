import re

import torch
from torch import nn

from pitch_detection_auto.configuration import Configuration
from pitch_detection_auto.pitch_det_net_v1 import PitchDetNet as PitchDetNetV1
from pitch_detection_auto.pitch_det_net_v2 import PitchDetNet as PitchDetNetV2
from pitch_detection_auto.pitch_det_net_v3 import PitchDetNet as PitchDetNetV3
from pitch_detection_auto.pitch_det_net_v4 import PitchDetNet as PitchDetNetV4
from pitch_detection_auto.synth_net_v1 import SynthNet as SynthNetV1
from pitch_detection_auto.synth_net_v2 import SynthNet as SynthNetV2

_PITCH_DET_REGISTRY = {
    1: PitchDetNetV1,
    2: PitchDetNetV2,
    3: PitchDetNetV3,
    4: PitchDetNetV4,
}

_SYNTH_REGISTRY = {
    1: SynthNetV1,
    2: SynthNetV2
}


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
        f = self.pitch_det_net(x)
        f_sigmoid = torch.sigmoid(f)
        s = self.synth(f_sigmoid)  # (B,1,F,T)
        return s, f_sigmoid


def entropy_term(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    - Add the channels
    - Considers (B,T) slices, as distributions over frequency
    - Calculates the entropy of these distributions, returning a (B,T) tensor
    a: (B,C,F,T)
    out: (B)
    """
    p = a.sum(dim=1, keepdim=False)  # (B,F,T)
    p = p / (p.sum(dim=-2, keepdim=True) + eps)
    h = -(p * (p + eps).log()).sum(dim=-2)  # (B,T)
    return h
