"""
Utility functions for genmusic.
"""

import random
from typing import List, Protocol, Sequence, Any


# =========================
# Randomness policies
# =========================


class RandomPolicy(Protocol):
    """Protocol for custom randomness policies."""

    def accept(self, p: float, *, rng: random.Random) -> bool: ...
    def choice(self, seq: Sequence[Any], *, rng: random.Random) -> Any: ...


class DefaultRandomPolicy:
    """Default randomness policy using Python's random module."""

    def accept(self, p: float, *, rng: random.Random) -> bool:
        p = max(0.0, min(1.0, float(p)))
        return rng.random() < p

    def choice(self, seq: Sequence[Any], *, rng: random.Random) -> Any:
        return rng.choice(list(seq))


class WeightedRandomPolicy:
    """Random policy with weighted choices."""

    def __init__(self, weights: List[float] = None):
        self._weights = weights

    def accept(self, p: float, *, rng: random.Random) -> bool:
        return rng.random() < p

    def choice(self, seq: Sequence[Any], *, rng: random.Random) -> Any:
        return rng.choice(list(seq))


# =========================
# Euclidean rhythm (Bjorklund)
# =========================


def bjorklund(k: int, n: int) -> List[int]:
    """
    Generate Euclidean rhythm pattern using Bjorklund algorithm.

    Args:
        k: Number of pulses (hits)
        n: Total number of steps

    Returns:
        List of 0s and 1s representing the pattern
    """
    if n <= 0:
        return []
    k = max(0, min(k, n))
    if k == 0:
        return [0] * n
    if k == n:
        return [1] * n

    pattern, counts, remainders = [], [], []
    divisor = n - k
    remainders.append(k)
    level = 0
    while True:
        counts.append(divisor // remainders[level])
        remainders.append(divisor % remainders[level])
        divisor = remainders[level]
        level += 1
        if remainders[level] <= 1:
            break
    counts.append(divisor)

    def build(level_: int) -> None:
        if level_ == -1:
            pattern.append(0)
        elif level_ == -2:
            pattern.append(1)
        else:
            for _ in range(counts[level_]):
                build(level_ - 1)
            if remainders[level_] != 0:
                build(level_ - 2)

    build(level)
    if 1 in pattern:
        first = pattern.index(1)
        pattern[:] = pattern[first:] + pattern[:first]
    return pattern


def euclidean_pattern(steps: int, pulses: int, rotation: int = 0) -> List[int]:
    """Generate a Euclidean pattern with optional rotation."""
    p = bjorklund(pulses, steps)
    if not p:
        return p
    r = rotation % len(p)
    return p[-r:] + p[:-r] if r else p


# =========================
# Probability helpers
# =========================


def coin_flip(p: float = 0.5, rng: random.Random = None) -> bool:
    """Simple coin flip with probability."""
    rng = rng or random.Random()
    return rng.random() < p


def choose_weighted(
    items: List[Any], weights: List[float], rng: random.Random = None
) -> Any:
    """Choose an item from a list with weighted probability."""
    rng = rng or random.Random()
    return rng.choices(items, weights=weights, k=1)[0]


def random_walk(
    current: float,
    step_size: float,
    min_val: float = -1.0,
    max_val: float = 1.0,
    rng: random.Random = None,
) -> float:
    """Generate a random walk value."""
    rng = rng or random.Random()
    value = current + rng.uniform(-step_size, step_size)
    return max(min_val, min(max_val, value))


# =========================
# Math helpers
# =========================


def midi_to_hz(midi: int) -> float:
    """Convert MIDI note number to Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def hz_to_midi(hz: float) -> int:
    """Convert Hz to MIDI note number."""
    return int(round(69 + 12 * (hz / 440.0).log2()))


def db_to_linear(db: float) -> float:
    """Convert decibels to linear gain."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear gain to decibels."""
    import math

    return 20.0 * math.log10(linear) if linear > 0 else -float("inf")


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def map_range(
    value: float,
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float,
) -> float:
    """Map a value from one range to another."""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
