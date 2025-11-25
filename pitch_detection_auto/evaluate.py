import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb


def log_autoencoder_sample(model, sample_specs, step):
    """Log original specs, f0 maps and reconstructions as one image."""
    if not len(sample_specs) == 4:
        raise ValueError("Expecting 4 samples")

    dev = next(model.parameters()).device
    with torch.no_grad():
        x = sample_specs.unsqueeze(1).float().to(dev)  # (4,1,F,T)
        y, f = model(x)

    x_np = x.squeeze().cpu().numpy()
    f_np = f.sum(dim=1).squeeze().cpu().numpy()
    y_np = y.squeeze().cpu().numpy()

    fig, ax = plt.subplots(3, 4, figsize=(6, 7), constrained_layout=True)
    for i in range(4):
        for im, title, a in zip((x_np[i], f_np[i], y_np[i]), ("original", "f0", "output"), ax[:, i]):
            a.imshow(im, aspect="auto", origin="lower", cmap="coolwarm")
            a.set_title(title)
            a.axis("off")

    wandb.log({"epoch_samples": [wandb.Image(fig)], "f0_min": f_np.min(), "f0_max": f_np.max()}, step=step)
    plt.close(fig)


def log_synth_kernels(model, step, hide_f0) -> None:
    """Log SynthNet kernels as a heat-map image.

    Kernels have shape (C,T,F) when T==1 and (C,F,T) when T>1.
    We keep frequency on the y-axis and flatten (channel,time) on x.
    """
    with torch.no_grad():
        ker = F.softplus(model.synth.kernel).cpu().squeeze(1)  # (C,·,·)
        if hide_f0:
            ker = ker[:, 1:]

    if len(ker.shape) == 2:  # (C,F) → 1d case
        img = ker.numpy().T  # (F,C)
        xlabel = "channel"
    elif len(ker.shape) == 3:  # (C,F,T) → 2d case
        C, F_, T = ker.shape
        img = ker.permute(1, 0, 2).reshape(F_, C * T).numpy()  # (F, C·T)
        xlabel = "channel · time"
    else:
        raise ValueError()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(img, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_title("SynthNet kernels")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency")
    ax.axis("auto")

    wandb.log(
        {"kernels": [wandb.Image(fig)],
         "kernels_min": img.min(),
         "kernels_max": img.max()},
        step=step,
    )
    plt.close(fig)


def log_pitch_det_sample(model, spec_tensor, current_step):
    """Log original spec, reconstruction as one image."""
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = spec_tensor.unsqueeze(0).unsqueeze(0).float().to(dev)  # (1,1,F,T)
        y = torch.sigmoid(model(x))

    x_np = x.squeeze().cpu().numpy()
    y_np = y[:, 0].squeeze().cpu().numpy()

    fig, ax = plt.subplots(2, 1, figsize=(6, 7), constrained_layout=True)
    for im, title, a in zip((x_np, y_np), ("original", "output"), ax):
        a.imshow(im, aspect="auto", origin="lower", cmap="viridis")
        a.set_title(title)
        a.axis("off")

    wandb.log({"epoch_samples": [wandb.Image(fig)]}, step=current_step)
    plt.close(fig)
