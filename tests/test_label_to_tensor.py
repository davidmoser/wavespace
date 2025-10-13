"""Tests for storing and loading :mod:`datasets.poly_dataset` artifacts."""

from __future__ import annotations

import torch

from pitch_detection_supervised.utils import label_to_tensor


def test_label_to_tensor() -> None:
    duration = 1.
    hop = 0.5
    centers = [100, 200, 500]

    # single event, match freq, match time
    event1 = [(100, 0.5, 1.)]
    result1 = label_to_tensor(event1, centers, duration, hop)
    expected1 = torch.tensor([[0., 1.], [0., 0.], [0., 0.]])
    torch.testing.assert_close(result1, expected1)
    # single event, between freq, match time
    event2 = [(150, 0.5, 1.)]
    result2 = label_to_tensor(event2, centers, duration, hop)
    expected2 = torch.tensor([[0., 0.5], [0., 0.5], [0., 0.]])
    torch.testing.assert_close(result2, expected2)
    # single event, match freq, between time
    event3 = [(500, 0.3, 0.6)]
    result3 = label_to_tensor(event3, centers, duration, hop)
    expected = torch.tensor([[0., 0.], [0., 0.], [0.2/0.5, 0.1/0.5]])
    torch.testing.assert_close(result3, expected)
    # single event, between freq, between time
    event4 = [(300, 0.2, 0.6)]
    result4 = label_to_tensor(event4, centers, duration, hop)
    time_split = torch.tensor([0.3 / 0.5, 0.1 / 0.5])
    fac_200 = (500-300)/(500-200)
    fac_500 = (300-200)/(500-200)
    expected4 = torch.tensor([[0., 0.], fac_200*time_split, fac_500*time_split ])
    torch.testing.assert_close(result4, expected4)
    # three events
    event5 = event1 + event2 + event4
    result5 = label_to_tensor(event5, centers, duration, hop)
    expected5 = expected1 + expected2 + expected4
    torch.testing.assert_close(result5, expected5)