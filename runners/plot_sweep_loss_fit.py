#!/usr/bin/env python3
"""
Fetch a Weights & Biases sweep, scatter-plot learning rate vs best validation loss for all runs,
fit a parabola in log10(LR) space, report the LR at the vertex (minimum), and overlay the fit.

Notes:
- LR is read from config keys: "lr" or "learning_rate" (override by setting lr_key).
- "best_val_loss" is read from run.summary if present; otherwise min over the run's history[val_metric].
- The quadratic fit is done on x = log10(lr). The reported LR* = 10**x_vertex.
"""

import math
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import wandb


def fit_parabola(lrs: np.ndarray, losses: np.ndarray, use_log_scaling: bool) -> Tuple[np.ndarray, float]:
    """
    Fit y ~ a*(log10(lr))^2 + b*(log10(lr)) + c. log10 optional
    Returns (coeffs a,b,c), x_vertex (in log10 space), lr_star (10**x_vertex).
    """
    if use_log_scaling:
        x = np.log10(lrs)
    else:
        x = lrs
    coeffs = np.polyfit(x, losses, deg=2)  # [a, b, c]
    a, b, c = coeffs
    if a == 0:
        idx = int(np.argmin(losses))
        return coeffs, lrs[idx]
    lr_star = -b / (2 * a)
    if use_log_scaling:
        lr_star = float(10 ** lr_star)
    return coeffs, lr_star


def plot_parameter_vs_loss(
        sweep_path: str,
        parameter_key: str,  # e.g., "lr" or "learning_rate"; None = auto-detect
        loss_key: str,
        out: str,
        use_log_scaling: bool,
        param_from_model_config: bool = False,
):
    api = wandb.Api()
    sweep = api.sweep(sweep_path)

    xs: List[float] = []
    ys: List[float] = []

    for run in sweep.runs:
        if run.state != "finished":
            continue
        if param_from_model_config:
            parameter = run.config["model_config"][parameter_key]
        else:
            parameter = run.config[parameter_key]
        if parameter is None or (parameter <= 0 and use_log_scaling) or not math.isfinite(parameter):
            continue
        loss = run.history(keys=[loss_key])[-20:][loss_key].mean()
        if loss is None or not math.isfinite(loss):
            continue
        xs.append(parameter)
        ys.append(loss)

    params = np.array(xs, dtype=float)
    losses = np.array(ys, dtype=float)

    coeffs, parameter_star = fit_parabola(params, losses, use_log_scaling)

    # Prepare smooth curve over observed LR range in log space
    if use_log_scaling:
        x_min, x_max = np.log10(params).min(), np.log10(params).max()
    else:
        x_min, x_max = params.min(), params.max()
    x_grid = np.linspace(x_min, x_max, 400)
    a, b, c = coeffs
    y_fit = a * x_grid ** 2 + b * x_grid + c
    if use_log_scaling:
        param_grid = 10 ** x_grid
    else:
        param_grid = x_grid

    # Plot
    param_text = f"log10({parameter_key})" if use_log_scaling else parameter_key
    plt.figure(figsize=(7, 5))
    plt.scatter(params, losses, label="runs", alpha=0.9)
    plt.plot(param_grid, y_fit, label=f"parabolic fit (in {param_text})", linewidth=2)
    if x_min <= parameter_star <= x_max:
        plt.axvline(parameter_star, linestyle="--", linewidth=1, label=f"{parameter_key}* = {parameter_star:.3e}")
    if use_log_scaling:
        plt.xscale("log")
    plt.xlabel(parameter_key)
    plt.ylabel("loss")
    plt.title(f"{parameter_key} vs loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)

    print(f"Estimated optimal {parameter_key} (vertex of quadratic in {param_text}): {parameter_star:.6g}")
    print(f"Coefficients (a, b, c) on y = a*({param_text})^2 + b*({param_text}) + c: {a:.6g}, {b:.6g}, {c:.6g}")
    print(f"Plot saved to: {out}")


if __name__ == "__main__":
    # plot_parameter_vs_loss(
    #     sweep_path="david-moser-ggg/pitch-detection-supervised/sweeps/9xq1acmj",
    #     parameter_key="lr",
    #     use_log_scaling=True,
    #     loss_key="train/loss",
    #     out="../results/pitch_detection_supervised/transformer_lr_loss_updates.png",
    # )
    plot_parameter_vs_loss(
        sweep_path="david-moser-ggg/pitch-detection-supervised/sweeps/9xq1acmj",
        parameter_key="dropout",
        use_log_scaling=False,
        loss_key="train/loss",
        out="../results/pitch_detection_supervised/sweep3_dropout_loss.png",
        param_from_model_config=True,
    )
    # plot_parameter_vs_loss(
    #     sweep_path="david-moser-ggg/pitch-detection-supervised/sweeps/9xq1acmj",
    #     parameter_key="weight_decay",
    #     use_log_scaling=False,
    #     loss_key="train/loss",
    #     out="../results/pitch_detection_supervised/sweep2_weight_decay_loss.png",
    # )
