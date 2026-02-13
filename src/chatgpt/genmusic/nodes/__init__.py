"""
Nodes module for genmusic.

Provides musical and timing nodes:
- Clock, EventScheduler: timing
- EuclideanRhythmNode: Euclidean rhythm patterns
- DrumNode: drum patterns
- Arpeggiator: melodic generation
- BassNode: bass line generation
- ContinuousSignal, Integrator: control signals
"""

from .base import Node, NodeGroup
from .timing import Clock, EventScheduler, EuclideanPattern
from .rhythm import EuclideanRhythmNode, DrumNode, DrumConfig
from .arp import Arpeggiator, ArpConfig
from .control import ContinuousSignal, Integrator
from .bass import BassNode

__all__ = [
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
]
