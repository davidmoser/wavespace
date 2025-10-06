from datasets.create_latent_store import create_latent_store
from datasets.sine_wave_dataset import RandomSineWaveDataset

duration = 1
sampling_rate = 44100
min_freq = 100
max_freq = 10000
num_samples = 1000

dataset = RandomSineWaveDataset(
    duration=duration,
    sampling_rate=sampling_rate,
    min_frequency=min_freq,
    max_frequency=max_freq,
    num_samples=num_samples,
)

create_latent_store(
    dataset,
    f"../resources/encodec_latents/sines_1",
    # sample_rate: Optional[int] = None,
    # map_size_bytes: int = 1 << 33,
    # device: Optional[torch.device] = None,
    # encoder: Optional["EncodecModel"] = None,
    # target_bandwidth: Optional[float] = None,
)
