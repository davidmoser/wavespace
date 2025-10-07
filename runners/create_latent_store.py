from datasets.create_latent_store import create_latent_store
from datasets.sine_wave_dataset import SineWaveDataset

duration = 1
sampling_rate = 24000
min_freq = 100
max_freq = 10000
num_samples = 10000

dataset = SineWaveDataset(
    duration=duration,
    sampling_rate=sampling_rate,
    min_frequency=min_freq,
    max_frequency=max_freq,
    num_samples=num_samples,
)

create_latent_store(
    dataset,
    dataset_path=f"../resources/encodec_latents/sines_1",
    dataset_sample_rate=sampling_rate,
    metadata={
        "dataset": "RandomSineWaveDataset",
        "parameters": {
            "duration": duration,
            "sampling_rate": sampling_rate,
            "min_frequency": min_freq,
            "max_frequency": max_freq,
            "num_samples": num_samples,
        },
    },
    # sample_rate: Optional[int] = None,
    # device: Optional[torch.device] = None,
    # encoder: Optional["EncodecModel"] = None,
    # target_bandwidth: Optional[float] = None,
)
