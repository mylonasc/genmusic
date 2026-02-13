"""
Instrument presets for synthesizers.
"""

from typing import Dict, Tuple

# Instrument presets: channel -> (bank, program)
# GM programs: https://www.midi.org/specifications/midi-reference-tables/gm-level-1-program-chart
INSTRUMENT_PRESETS: Dict[str, Dict[int, Tuple[int, int]]] = {
    "acoustic": {
        0: (0, 0),  # Acoustic Grand
        1: (0, 33),  # Electric Bass (finger)
        2: (0, 48),  # Strings
        9: (0, 0),  # Standard Drum Kit
    },
    "electronic": {
        0: (0, 19),  # Synth Piano
        1: (0, 38),  # Synth Bass
        2: (0, 80),  # Sawtooth Lead
        9: (0, 17),  # Analog Drums
    },
    "303": {
        0: (0, 86),  # Lead Square (303-like)
        1: (0, 39),  # Synth Bass
        2: (0, 17),  # Timpani (for sub)
        9: (0, 17),  # Analog Drums
    },
    "techno": {
        0: (0, 18),  # Rock Organ
        1: (0, 40),  # Fretless Bass
        2: (0, 43),  # Synth Strings
        9: (0, 17),  # Analog Drums
    },
    "jazz": {
        0: (0, 1),  # Bright Acoustic Piano
        1: (0, 32),  # Acoustic Bass
        2: (0, 52),  # Choir Aahs
        9: (0, 0),  # Standard Drum Kit
    },
    "ambient": {
        0: (0, 88),  # Pad Sweep
        1: (0, 39),  # Synth Bass
        2: (0, 89),  # Pad Halo
        9: (0, 17),  # Analog Drums
    },
}


def get_preset(name: str) -> Dict[int, Tuple[int, int]]:
    """Get an instrument preset by name."""
    return INSTRUMENT_PRESETS.get(name, INSTRUMENT_PRESETS["acoustic"])
