"""
Musical scales and harmony system.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Scale:
    """Musical scale defined by intervals within an octave."""

    name: str
    intervals: Tuple[int, ...]
    octave: int = 12

    def degree_to_semitones(self, degree: int) -> int:
        """Convert scale degree to MIDI semitones."""
        n = len(self.intervals)
        if n == 0:
            return 0
        octave_shift, idx = divmod(degree, n)
        return self.intervals[idx] + octave_shift * self.octave

    def degree_to_note(self, degree: int, root: int = 60) -> int:
        """Convert scale degree to MIDI note number."""
        return root + self.degree_to_semitones(degree)

    def scale_notes(self, root: int = 60, octaves: int = 1) -> List[int]:
        """Get all notes in the scale across octaves."""
        notes = []
        for octv in range(octaves):
            base = root + (octv * self.octave)
            for deg in range(len(self.intervals)):
                notes.append(base + self.intervals[deg])
        return notes


class ScaleRegistry:
    """Registry for managing musical scales."""

    def __init__(self):
        self._scales: Dict[str, Scale] = {}

    def add(self, scale: Scale) -> None:
        """Register a scale."""
        self._scales[scale.name] = scale

    def get(self, name: str) -> Scale:
        """Get a scale by name."""
        return self._scales[name]

    def names(self) -> List[str]:
        """List all registered scale names."""
        return sorted(self._scales.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._scales


# Standard scale definitions
SCALES = {
    "major": Scale("major", (0, 2, 4, 5, 7, 9, 11)),
    "minor": Scale("minor", (0, 2, 3, 5, 7, 8, 10)),
    "dorian": Scale("dorian", (0, 2, 3, 5, 7, 9, 10)),
    "phrygian": Scale("phrygian", (0, 1, 3, 5, 7, 8, 10)),
    "lydian": Scale("lydian", (0, 2, 4, 6, 7, 9, 11)),
    "mixolydian": Scale("mixolydian", (0, 2, 4, 5, 7, 9, 10)),
    "chromatic": Scale("chromatic", tuple(range(12))),
    "whole": Scale("whole", (0, 2, 4, 6, 8, 10)),
    "diminished": Scale("diminished", (0, 3, 6, 9)),
    "augmented": Scale("augmented", (0, 4, 8)),
    "pentatonic_major": Scale("pentatonic_major", (0, 2, 4, 7, 9)),
    "pentatonic_minor": Scale("pentatonic_minor", (0, 3, 5, 7, 10)),
    "blues": Scale("blues", (0, 3, 5, 6, 7, 10)),
    "harmonic_minor": Scale("harmonic_minor", (0, 2, 3, 5, 7, 8, 11)),
    "melodic_minor": Scale("melodic_minor", (0, 2, 3, 5, 7, 9, 11)),
    "bebop_dominant": Scale("bebop_dominant", (0, 2, 4, 5, 7, 9, 10, 11)),
    "spanish_phrygian": Scale("spanish_phrygian", (0, 1, 4, 5, 7, 8, 10)),
}


def default_scales() -> ScaleRegistry:
    """Create a scale registry with standard scales."""
    reg = ScaleRegistry()
    for scale in SCALES.values():
        reg.add(scale)
    return reg


# Chord definitions (root, intervals)
CHORDS = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
    "augmented": (0, 4, 8),
    "sus4": (0, 5, 7),
    "sus2": (0, 2, 7),
    "major7": (0, 4, 7, 11),
    "minor7": (0, 3, 7, 10),
    "dominant7": (0, 4, 7, 10),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "9": (0, 4, 7, 10, 14),
    "add9": (0, 4, 7, 14),
}


def chord_notes(root: int, chord_name: str, octaves: int = 1) -> List[int]:
    """Get MIDI notes for a chord."""
    intervals = CHORDS.get(chord_name, (0, 4, 7))
    notes = []
    for _ in range(octaves):
        for interval in intervals:
            notes.append(root + interval)
        root += 12
    return notes
