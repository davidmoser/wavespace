import math

import torch


def label_to_tensor(
        events,  # list of (frequency_hz, onset_s, offset_s)
        frequency_bin_centers_hz,  # list/1D array-like
        duration_s: float,
        frame_hop_s: float,
        device: torch.device = None,
        dtype=torch.float32,
):
    # Prepare axes
    f_centers = torch.as_tensor(frequency_bin_centers_hz, dtype=dtype, device=device)
    assert f_centers.ndim == 1, "frequency_bin_centers_hz must be 1D"
    n_f = f_centers.numel()
    n_frames = int(math.ceil(duration_s / frame_hop_s))
    y = torch.zeros(n_f, n_frames, dtype=dtype, device=device)

    # Helper: interpolate a scalar frequency onto neighboring freq bins (handles non-uniform centers)
    def freq_interp_weights(f):
        if f <= f_centers[0]:
            return [(0, 1.0)]
        if f >= f_centers[-1]:
            return [(n_f - 1, 1.0)]
        # searchsorted for right neighbor
        r = int(torch.searchsorted(f_centers, torch.tensor(f, dtype=dtype, device=device)).item())
        l = r - 1
        denom = (f_centers[r] - f_centers[l]).clamp_min(torch.finfo(dtype).eps)
        alpha = float((f - float(f_centers[l])) / float(denom))
        return [(l, 1.0 - alpha), (r, alpha)]

    # Rasterize each event with bilinear (freq × time) interpolation
    for f_hz, t_on, t_off in events:
        # Clamp times to [0, duration]
        t0 = max(0.0, min(float(t_on), duration_s))
        t1 = max(0.0, min(float(t_off), duration_s))
        if not (t1 > t0):
            continue

        # Frequency weights (up to two neighbors)
        f_w = freq_interp_weights(float(f_hz))

        # Time coverage (fractional coverage per frame)
        s = t0 / frame_hop_s
        e = t1 / frame_hop_s
        i_start = max(0, int(math.floor(s)))
        i_end = min(n_frames - 1, int(math.ceil(e)) - 1)
        if i_end < i_start:
            # Entire event falls within a single fractional frame index region
            i_start = i_end = min(n_frames - 1, max(0, int(math.floor((s + e) * 0.5))))

        # Compute overlap per frame (vectorized over the [i_start, i_end] window)
        idx = torch.arange(i_start, i_end + 1, device=device, dtype=dtype)
        # Frame index interval is [i, i+1) in index space
        left = torch.clamp(torch.minimum(torch.maximum(torch.tensor(s, device=device, dtype=dtype), idx), idx + 1.0),
                           min=0.0)
        right = torch.clamp(torch.minimum(torch.maximum(torch.tensor(e, device=device, dtype=dtype), idx), idx + 1.0),
                            min=0.0)
        overlap = (right - left).clamp(min=0.0, max=1.0)  # fractional time weight per frame

        if overlap.numel() == 0:
            continue

        # Add bilinear contributions
        for fi, fw in f_w:
            y[fi, i_start: i_end + 1] += fw * overlap

    # Optionally cap at 1.0 if desired for overlapping events:
    # y.clamp_(0.0, 1.0)

    return y
