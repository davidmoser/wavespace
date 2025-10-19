"""Dilated temporal convolutional network for pitch classification."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DilatedTCN(nn.Module):
    """Dilated temporal convolutional network for pitch estimation.

    Args:
        n_classes: Number of pitch classes ``F``.
        seq_len: Expected sequence length ``T``.
        latent_dim: Dimensionality ``L`` of Encodec latents.
        hidden_dim: Hidden channel size ``H`` for internal convolutions.
        use_third_block: Whether to include the third dilated block.
        dropout: Dropout probability applied after each GELU activation.

    Input Tensor:
        ``x`` of shape ``[B, L, T]`` containing the per-frame Encodec latents.

    Output Tensor:
        Logits of shape ``[B, F, T]`` giving the per-step pitch class scores.
    """

    def __init__(
            self,
            *,
            n_classes: int = 128,
            seq_len: int = 75,
            latent_dim: int = 128,
            hidden_dim: int = 256,
            use_third_block: bool = False,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        self.n_classes = n_classes
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_third_block = use_third_block
        self.dropout = dropout

        blocks: List[nn.Sequential] = [
            self._make_block(latent_dim, hidden_dim, kernel_size=9, dilation=1, padding=4, dropout=dropout)
        ]
        blocks.append(
            self._make_block(hidden_dim, hidden_dim, kernel_size=9, dilation=2, padding=8, dropout=dropout)
        )
        if use_third_block:
            blocks.append(
                self._make_block(hidden_dim, hidden_dim, kernel_size=9, dilation=4, padding=16, dropout=dropout)
            )

        self.conv_blocks = nn.Sequential(*blocks)
        self.classifier = nn.Conv1d(hidden_dim, n_classes, kernel_size=1)

    @staticmethod
    def _make_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            padding: int,
            dropout: float,
    ) -> nn.Sequential:
        layers: List[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.BatchNorm1d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"DilatedTCN expected a 3D tensor with shape [B, L, T], but received {tuple(x.shape)}")
        batch_size, latent_dim, seq_len = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Input sequence length {seq_len} does not match expected seq_len={self.seq_len}.")
        if latent_dim != self.latent_dim:
            raise ValueError(
                f"Input latent dimension {latent_dim} does not match expected latent_dim={self.latent_dim}."
            )

        x = self.conv_blocks(x)

        logits = self.classifier(x)

        expected_shape = (batch_size, self.n_classes, self.seq_len)
        if logits.shape != expected_shape:
            raise RuntimeError(
                "DilatedTCN produced unexpected output shape: "
                f"got {tuple(logits.shape)}, expected {expected_shape}."
            )

        return logits
