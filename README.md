# Wavespace

**Overview**

* Self-education project on audio ML
* Explore datasets, training setups, SOTA methods
* Focus on pitch salience and piano-roll prediction
* Build end-to-end pipelines for experiments

**Key findings**

* Unsupervised training: Pitch salience prediction in latent space is difficult, because of the delicate tradeoff
  between latent space control and autoencoding fidelity.
* Supervised training: Conv models are often stronger than transformers. Transformers don't learn time dependent
  structures well in spectrograms, because consecutive frames are very close together as vectors.
* Automated tooling with YAML configs, W&B and parallel Runpod workers speeds up experiment cycle

**Tooling**

* One click scripts for local and remote runs
* W&B for tracking and sweeps
* Runpod for training on multiple workers
* GitHub and Codex workflows for iteration

## Shared Infrastructure

* Per-experiment packages hold model code, **Python** and **Torch**
* Python scripts trigger runs and sweeps
* **YAML configs** sent to **Weights & Biases (W&B)**
* Local or **Runpod** worker pulls config from W&B and trains
* Training streams losses and samples to W&B
* Post-processing scripts pull sweep data and compute fits
* **Docker** image for Runpod workers
* Datasets stored locally and on a Runpod volume

```mermaid
graph LR
    L[Local]
    W[W&B]
    R[Runpod]

    L -->|"run/sweep configuration"| W
    L -->|"start run/sweep job"| R
    W -->|"run/sweep configuration"| R
    R -->|"results (loss, samples)"| W
    W -->|"results (loss, samples), loss fits"| L
```

## Projects

### Autoencoding of spectrograms with UNet and ConvNet

#### Abstract

This project explores spectrogram autoencoding using UNet variants and both 1D and 2D convolutional architectures. The 
aim was to achieve high-fidelity reconstruction while gaining practical experience with spectrogram preprocessing, audio 
datasets, and neural architectures. Hyperparameter sweeps covered channel counts, learning rates, and batch sizes.

#### Methods

* Dataset: Medley Solos DB, 21k samples, 2s duration, solos of 7 instruments & singer, 11 GB
* Samples: log-frequency spectrograms and melspectrograms, 256 frequency bins, 173 time bins
* Architectures: UNet, 1D- and 2D- ConvNets, up to 1024 channels

#### Results

UNet models produced strong fidelity, down to a L1-loss of 0.77, likely due to their skip-connection bypass paths, while pure ConvNets achieved moderate
but consistent reconstruction quality.

### Pitch salience with UNet and convolution self supervised

#### Abstract

#### Methods

* Dataset: Medley Solos DB, 21k samples, 2s duration, solos of 7 instruments & singer, 11 GB
* Dataset: Maestro Dataset, piano recordings with Midi labels, 129 GB, processed to 10k chunks, 20s each, only samples

#### Results

### Pitch salience with Encodec latents and transformer supervised

#### Abstract

#### Methods

* Dataset: Synthetic non-Midi-pitch polyphonic samples, mixing pitches, durations, envelopes, timbres, 1 GB
* Dataset: Maestro Dataset, piano recordings with Midi labels, 129 GB, processed to 50k chunks, 10s each

#### Results

