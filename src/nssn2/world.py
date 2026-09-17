from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorldTape:
    events: np.ndarray
    labels: np.ndarray
    prototypes: np.ndarray


def make_world_tape(
    seed: int,
    episodes: int,
    sensors: int,
    causes: int,
    noise: float,
) -> WorldTape:
    if episodes <= 0 or sensors <= 0 or causes <= 1:
        raise ValueError("episodes/sensors must be positive and causes must exceed one")
    if noise < 0.0:
        raise ValueError("noise must be non-negative")

    rng = np.random.default_rng(seed)

    raw = rng.uniform(0.08, 0.92, size=(causes, sensors))
    # Give each latent cause a few stronger anchor sensors while preserving overlap.
    for cause in range(causes):
        anchor = (cause * max(1, sensors // causes)) % sensors
        width = max(1, sensors // (2 * causes))
        indices = (anchor + np.arange(width)) % sensors
        raw[cause, indices] = np.clip(raw[cause, indices] + 0.35, 0.0, 1.0)
    prototypes = raw.astype(float)

    labels = np.tile(np.arange(causes, dtype=int), int(np.ceil(episodes / causes)))[:episodes]
    # Shuffle within balanced blocks so labels remain balanced without a trivial periodic order.
    block = max(causes * 2, causes)
    for start in range(0, episodes, block):
        stop = min(start + block, episodes)
        rng.shuffle(labels[start:stop])

    events = prototypes[labels] + rng.normal(0.0, noise, size=(episodes, sensors))
    events = np.clip(events, 0.0, 1.0)

    return WorldTape(events=events.astype(float), labels=labels, prototypes=prototypes)
