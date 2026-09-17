import numpy as np

from nssn2.world import make_world_tape


def test_world_tape_is_deterministic_balanced_and_bounded():
    first = make_world_tape(seed=7, episodes=24, sensors=8, causes=3, noise=0.08)
    second = make_world_tape(seed=7, episodes=24, sensors=8, causes=3, noise=0.08)
    other = make_world_tape(seed=8, episodes=24, sensors=8, causes=3, noise=0.08)

    assert first.events.shape == (24, 8)
    assert first.labels.shape == (24,)
    assert np.array_equal(first.events, second.events)
    assert np.array_equal(first.labels, second.labels)
    assert not np.array_equal(first.events, other.events)
    assert np.all((first.events >= 0.0) & (first.events <= 1.0))
    assert set(first.labels.tolist()) == {0, 1, 2}

    counts = np.bincount(first.labels, minlength=3)
    assert counts.max() - counts.min() <= 1
