from datasets.create_latent_store import create_latent_store
from datasets.poly_dataset import PolyphonicAsyncDataset

duration = 2.0
sampling_rate = 24000
min_freq = 50.0
max_freq = 5000.0
num_samples = 10000
max_polyphony = 10
min_note_duration = 0.12
label_type = "activation"

dataset = PolyphonicAsyncDataset(
    n_samples=num_samples,
    freq_range=(min_freq, max_freq),
    max_polyphony=max_polyphony,
    sr=sampling_rate,
    duration=duration,
    min_note_duration=min_note_duration,
    label_type=label_type,
)

create_latent_store(
    dataset,
    dataset_path="../../resources/encodec_latents/poly_async_activation",
    metadata={
        "dataset": dataset.__class__.__name__,
        "parameters": {
            "duration": duration,
            "sampling_rate": sampling_rate,
            "min_frequency": min_freq,
            "max_frequency": max_freq,
            "num_samples": num_samples,
            "max_polyphony": max_polyphony,
            "min_note_duration": min_note_duration,
            "label_type": label_type,
        },
    },
    # sample_rate: Optional[int] = None,
    # device: Optional[torch.device] = None,
    # encoder: Optional["EncodecModel"] = None,
    # target_bandwidth: Optional[float] = None,
)
