import torch

# configuration
num_samples = 1000  # number of samples
num_frames = 200    # time frames per sample
num_bins = 256      # frequency bins
f_delta = 30        # f1 - f0 distance
magnitude = 100     # value at f0 and f1
save_file = "../resources/toy_spectra.pt"


if __name__ == "__main__":
    specs = torch.zeros(num_samples, num_bins, num_frames)
    for i in range(num_samples):
        f0 = torch.randint(0, num_bins - f_delta, (1,)).item()
        f1 = f0 + f_delta
        specs[i, f0] = magnitude
        specs[i, f1] = magnitude
    torch.save(specs, save_file)
    print(f"saved {num_samples} samples to {save_file}")
