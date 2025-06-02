import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from spectrogram_autoencoder.models import get_model


def show_kernels_3x3(model: nn.Module, max_cols: int = 8, cmap: str = "coolwarm"):
    """
    Visualise every 3×3 Conv2d kernel in the model as a small image, grouped by layer.
    The weight file must already be loaded into `model`.
    """
    # --- collect only 3×3 conv layers ------------------------------------------------
    target_layers = {name: m for name, m in model.named_modules()
                     if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)}

    for name, conv in target_layers.items():
        W = conv.weight.data.cpu()              # shape (out_ch, in_ch, 3, 3)
        W = W.mean(dim=1)                       # → (out_ch, 3, 3)  – average over input channels
        n_k, _, _ = W.shape

        n_cols      = min(max_cols, n_k)
        n_rows      = math.ceil(n_k / n_cols)
        fig, axes   = plt.subplots(n_rows, n_cols,
                                   figsize=(1.4 * n_cols, 1.4 * n_rows),
                                   squeeze=False)

        vmin, vmax  = W.min().item(), W.max().item()
        for k, ax in enumerate(axes.flat):
            if k < n_k:
                ax.imshow(W[k], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f"{name}  –  {n_k} kernels", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


version="v4"

model = get_model(version, base_ch=16)
model.load_state_dict(torch.load("../resources/checkpoints/spec_auto_v4.pt"))
show_kernels_3x3(model)
