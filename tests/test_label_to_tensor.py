"""Tests for storing and loading :mod:`datasets.poly_dataset` artifacts."""


import torch

from pitch_detection_supervised.utils import events_to_active_label


def test_events_to_active_label() -> None:
    duration = 1.
    no_frames = 2
    centers = [100, 200, 500]

    # single event, match freq, match time
    event1 = [(100, 0.5, 1.)]
    result1 = events_to_active_label(event1, centers, duration, no_frames)
    expected1 = torch.tensor([[0., 1.], [0., 0.], [0., 0.]])
    torch.testing.assert_close(result1, expected1)
    # single event, between freq, match time
    event2 = [(150, 0.5, 1.)]
    result2 = events_to_active_label(event2, centers, duration, no_frames)
    expected2 = torch.tensor([[0., 0.5], [0., 0.5], [0., 0.]])
    torch.testing.assert_close(result2, expected2)
    # single event, match freq, between time
    event3 = [(500, 0.3, 0.6)]
    result3 = events_to_active_label(event3, centers, duration, no_frames)
    expected = torch.tensor([[0., 0.], [0., 0.], [0.2/0.5, 0.1/0.5]])
    torch.testing.assert_close(result3, expected)
    # single event, between freq, between time
    event4 = [(300, 0.2, 0.6)]
    result4 = events_to_active_label(event4, centers, duration, no_frames)
    time_split = torch.tensor([0.3 / 0.5, 0.1 / 0.5])
    fac_200 = (500-300)/(500-200)
    fac_500 = (300-200)/(500-200)
    expected4 = torch.tensor([[0., 0.], fac_200*time_split, fac_500*time_split ])
    torch.testing.assert_close(result4, expected4)
    # three events
    event5 = event1 + event2 + event4
    result5 = events_to_active_label(event5, centers, duration, no_frames)
    expected5 = expected1 + expected2 + expected4
    torch.testing.assert_close(result5, expected5)
