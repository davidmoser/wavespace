import torch


def normalize_samples(samples: torch.Tensor) -> torch.Tensor:
    # samples = torch.log(samples)
    mins = samples.amin(dim=(2,), keepdim=True)
    maxs = samples.amax(dim=(2,), keepdim=True)
    samples = (samples - mins) / (maxs - mins + 1e-4)
    return samples
