"""Lightweight local-context MLP for supervised pitch detection."""

from torch import Tensor, nn


class LocalContextMLP(nn.Module):
    """Predict pitch class logits using local temporal context.

    Args:
        n_classes: Number of output pitch classes ``C``.
        seq_len: Expected number of time steps ``T`` in the input sequence.
        latent_dim: Dimensionality of the input latent features ``D``.
        hidden_dim: Hidden dimensionality ``H`` of the time-distributed MLP.
        kernel_size: Kernel size for the depthwise temporal convolution.
        dropout: Dropout probability used inside the MLP head.

    Input shape:
        ``(batch, time, latent_dim)``.

    Output shape:
        ``(batch, time, n_classes)`` containing the logits for each class.
    """

    def __init__(
            self,
            *,
            n_classes: int = 128,
            seq_len: int = 75,
            latent_dim: int = 128,
            hidden_dim: int = 256,
            kernel_size: int = 15,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd to preserve sequence length with symmetric padding."
            )
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if n_classes <= 0:
            raise ValueError("n_classes must be positive.")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in the interval [0, 1).")

        self.n_classes = n_classes
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dropout = dropout

        padding = kernel_size // 2

        self.temporal_conv = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=latent_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            groups=latent_dim,
            bias=True,
        )
        self.temporal_activation = nn.GELU()

        self.mlp_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the local context MLP.

        Args:
            x: Input tensor of shape ``(batch, latent_dim, time)``.

        Returns:
            Logits tensor of shape ``(batch, n_classes, time)``.
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected input tensor with 3 dimensions (batch, latent_time, time), got shape {tuple(x.shape)}."
            )
        batch_size, feature_dim, time_steps = x.shape
        if time_steps != self.seq_len:
            raise ValueError(f"Expected time dimension T={self.seq_len}, but received T={time_steps}.")
        if feature_dim != self.latent_dim:
            raise ValueError(f"Expected feature dimension D={self.latent_dim}, but received D={feature_dim}.")

        # Depthwise temporal convolution over local context.
        x = self.temporal_conv(x)
        x = self.temporal_activation(x)

        # Time-distributed MLP head operating on each time step independently.
        x = x.transpose(1, 2)
        logits = self.mlp_head(x)
        logits = logits.transpose(1, 2)

        if logits.shape != (batch_size, self.n_classes, time_steps):
            raise RuntimeError(
                "Unexpected logits shape after MLP head: "
                f"expected {(batch_size, self.n_classes, time_steps)}, got {tuple(logits.shape)}."
            )
        return logits
