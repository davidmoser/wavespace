import torch

from datasets.tensor_store import TensorStore

store = TensorStore(
    store_path="../../resources/encodec_latents/maestro_power_10000sam_20sec",
    transpose_labels=True,
)

sample, label = store[16]

hist, bin_edges = torch.histogram(label, bins=10)
print(torch.round(hist / hist.sum(), decimals=4))
print(torch.round(bin_edges, decimals=2))
