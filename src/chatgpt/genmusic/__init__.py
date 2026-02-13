"""
genmusic - Generative Algorithmic Music Control Framework

A modular, event-driven framework for generative music with asyncio.

Quick Start:
    from genmusic import System, EventBus, Clock, Arpeggiator
    from genmusic.scales import default_scales
    from genmusic.nodes import EuclideanRhythmNode, EuclideanPattern
    from genmusic.synth import FluidSynthSink

Architecture:
    - events: Event classes and EventBus (pub/sub)
    - nodes: Musical and timing nodes
    - scales: Musical scales and harmony
    - synth: Audio output sinks
    - utils: Utilities (rhythm, randomness, math)
    - system: System runner
"""

# Events
from .events import (
    Event,
    TickEvent,
    TriggerEvent,
    ValueEvent,
    ParamChangeEvent,
    NoteEvent,
    EventBus,
    EventFilter,
    Subscription,
    EventChain,
)

# Scales
from .scales import (
    Scale,
    ScaleRegistry,
    default_scales,
    CHORDS,
    chord_notes,
)

# Nodes
from .nodes import (
    Node,
    NodeGroup,
    Clock,
    EventScheduler,
    EuclideanPattern,
    EuclideanRhythmNode,
    DrumNode,
    DrumConfig,
    Arpeggiator,
    ArpConfig,
    ContinuousSignal,
    Integrator,
    BassNode,
)

# Utils
from .utils import (
    RandomPolicy,
    DefaultRandomPolicy,
    bjorklund,
    euclidean_pattern,
)

# Synth
from .synth import (
    FluidSynthSink,
    INSTRUMENT_PRESETS,
    get_preset,
)

# System
from .system import System


__version__ = "0.2.0"

__all__ = [
    # Events
    "Event",
    "TickEvent",
    "TriggerEvent",
    "ValueEvent",
    "ParamChangeEvent",
    "NoteEvent",
    "EventBus",
    "EventFilter",
    "Subscription",
    "EventChain",
    # Scales
    "Scale",
    "ScaleRegistry",
    "default_scales",
    "CHORDS",
    "chord_notes",
    # Nodes
    "Node",
    "NodeGroup",
    "Clock",
    "EventScheduler",
    "EuclideanPattern",
    "EuclideanRhythmNode",
    "DrumNode",
    "DrumConfig",
    "Arpeggiator",
    "ArpConfig",
    "ContinuousSignal",
    "Integrator",
    "BassNode",
    # Utils
    "RandomPolicy",
    "DefaultRandomPolicy",
    "bjorklund",
    "euclidean_pattern",
    # Synth
    "FluidSynthSink",
    "INSTRUMENT_PRESETS",
    "get_preset",
    # System
    "System",
]
