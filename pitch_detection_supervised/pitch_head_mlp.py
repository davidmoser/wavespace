"""Pitch detection module for classifying fundamental frequency from Encodec latents."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PitchHeadMLP(nn.Module):
    """Pitch classifier head using temporal pooling followed by an MLP.

    Args:
        n_classes: Number of fundamental frequency classes to predict.
        seq_len: Expected sequence length (number of frames) in the input.
        latent_dim: Dimensionality of latent vectors per frame.
        pool_kernel: Kernel size (in frames) for temporal average pooling.
        pool_stride: Stride (in frames) for temporal average pooling.
        dropout_p: Dropout probability applied within the MLP.

    Shapes:
        * Input: ``(batch, time, latent_dim)`` where ``time == seq_len``.
        * Output: ``(batch, n_classes)`` logits.
    """

    def __init__(
        self,
        n_classes: int = 128,
        seq_len: int = 75,
        latent_dim: int = 128,
        pool_kernel: int = 15,
        pool_stride: int = 15,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride

        self._pool_expected_frames = self._compute_pooled_frames(seq_len)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, n_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(module.bias, -bound, bound)

    def _compute_pooled_frames(self, seq_len: int) -> int:
        if seq_len < self.pool_kernel:
            raise ValueError(
                "Sequence length must be at least as large as the pooling kernel size."
            )
        pooled_frames = 1 + (seq_len - self.pool_kernel) // self.pool_stride
        if (seq_len - self.pool_kernel) % self.pool_stride != 0:
            raise ValueError(
                "Sequence length must evenly divide into pooling windows with the given stride."
            )
        return pooled_frames

    def forward(self, x: Tensor) -> Tensor:
        """Run the classifier head on Encodec latent representations.

        Args:
            x: Input tensor of shape ``(batch, time, latent_dim)``.

        Returns:
            Logits tensor of shape ``(batch, n_classes)``.
        """

        if x.dim() != 3:
            raise ValueError("Input tensor must have shape (batch, time, latent_dim).")
        batch_size, time, dim = x.shape
        if time != self.seq_len:
            raise AssertionError(
                f"Expected sequence length {self.seq_len}, but received {time}."
            )
        if dim != self.latent_dim:
            raise AssertionError(
                f"Expected latent dimension {self.latent_dim}, but received {dim}."
            )

        normalized = F.normalize(x, p=2.0, dim=-1, eps=1e-12)
        pooled = self._temporal_pool(normalized)
        pooled = pooled.mean(dim=1)
        logits = self.classifier(pooled)
        if logits.shape != (batch_size, self.n_classes):
            raise AssertionError("Output logits shape mismatch.")
        return logits

    def _temporal_pool(self, x: Tensor) -> Tensor:
        batch_size, time, _ = x.shape
        expected_frames = self._pool_expected_frames
        x_t = x.transpose(1, 2)
        pooled = F.avg_pool1d(x_t, kernel_size=self.pool_kernel, stride=self.pool_stride)
        pooled = pooled.transpose(1, 2)
        if pooled.shape[0] != batch_size or pooled.shape[1] != expected_frames:
            raise AssertionError(
                "Temporal pooling produced an unexpected output shape."
            )
        return pooled


if __name__ == "__main__":
    torch.manual_seed(0)
    model = PitchHeadMLP()
    dummy_input = torch.randn(4, 75, 128)
    output = model(dummy_input)
    print("Output shape:", output.shape)
