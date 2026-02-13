"""
Synth module for genmusic.

Provides audio output sinks:
- FluidSynthSink: Software synth output
"""

from .fluidsynth import FluidSynthSink
from .presets import INSTRUMENT_PRESETS, get_preset

__all__ = [
    "FluidSynthSink",
    "INSTRUMENT_PRESETS",
    "get_preset",
]
