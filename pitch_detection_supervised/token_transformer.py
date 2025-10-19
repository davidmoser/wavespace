"""Token Transformer module for pitch detection."""
from __future__ import annotations

import torch
import torch.nn as nn


class TokenTransformer(nn.Module):
    """Transformer-based classifier for Encodec latent tokens.

    Args:
        n_classes: Number of output classes per token.
        seq_len: Maximum sequence length supported by the positional embedding.
        latent_dim: Dimensionality of the input token features.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        ffn_dim: Dimension of the feed-forward network in the Transformer.
        dropout: Dropout rate applied within the Transformer.

    Shapes:
        * x: ``(batch, time, latent_dim)``
        * output: ``(batch, time, n_classes)``
    """

    def __init__(
            self,
            *,
            n_classes: int = 128,
            seq_len: int = 75,
            latent_dim: int = 128,
            d_model: int = 256,
            nhead: int = 4,
            num_layers: int = 2,
            ffn_dim: int = 512,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_classes <= 0:
            raise ValueError("n_classes must be positive")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if nhead <= 0:
            raise ValueError("nhead must be positive")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")

        self.n_classes = n_classes
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.d_model = d_model

        self.input_projection = nn.Linear(latent_dim, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the TokenTransformer.

        Args:
            x: Input tensor of shape ``(batch, time, latent_dim)``.

        Returns:
            Tensor of shape ``(batch, time, n_classes)`` containing class logits.
        """

        if x.dim() != 3:
            raise ValueError(
                f"Expected a 3D tensor for x of shape (batch, time, latent_dim), got {tuple(x.shape)}"
            )

        _, time_steps, feature_dim = x.shape
        if feature_dim != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, but received {feature_dim}"
            )
        if time_steps > self.seq_len:
            raise ValueError(
                f"Input sequence length {time_steps} exceeds maximum supported length {self.seq_len}"
            )

        projected = self.input_projection(x)

        position_ids = torch.arange(time_steps, device=x.device)
        position_embeddings = self.position_embedding(position_ids)
        encoded = projected + position_embeddings.unsqueeze(0)

        encoded = self.encoder(encoded)
        logits = self.classifier(encoded)
        return logits
