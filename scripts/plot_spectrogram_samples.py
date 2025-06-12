import os
import random
import matplotlib.pyplot as plt
import torch

# Path to the tensor file containing the spectrogram dataset
# Adjust this path to your environment
data_file = "../resources/logspectrograms.pt"

# Number of random samples to plot
n = 100

# Directory in which the output images will be saved
output_folder = "../resources/logspectrograms"


def main() -> None:
    if os.path.exists(output_folder):
        raise FileExistsError(f"Output folder '{output_folder}' already exists")
    os.makedirs(output_folder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(data_file, mmap=True, map_location=device)

    if n > len(data):
        raise ValueError(f"Requested {n} samples, but dataset only has {len(data)}")

    indices = random.sample(range(len(data)), n)

    for idx in indices:
        spec = data[idx]
        spec_np = spec.cpu().numpy()
        h, w = spec_np.shape
        plt.figure(figsize=(w / 100, h / 100))
        plt.imshow(spec_np, aspect="auto", origin="lower", cmap="coolwarm")
        plt.axis("off")
        plt.tight_layout(pad=0)
        out_path = os.path.join(output_folder, f"frame_{idx}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()


if __name__ == "__main__":
    main()
