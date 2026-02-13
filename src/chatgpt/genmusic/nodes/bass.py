"""
Bass node.
"""

import asyncio
import random
from typing import Optional

from ..events import EventBus, TriggerEvent, ParamChangeEvent, NoteEvent
from ..scales import ScaleRegistry
from .base import Node


class BassNode(Node):
    """Generates bass notes from triggers."""

    PATTERNS = ["root", "octave", "fifth", "walk"]

    def __init__(
        self,
        bus: EventBus,
        scales: ScaleRegistry,
        *,
        name: str = "bass",
        trigger_name: str = "bass_trigger",
        scale_name: str = "dorian",
        root_midi: int = 36,
        pattern: str = "octave",
        probability: float = 1.0,
        gate_beats: float = 0.5,
        velocity: float = 0.7,
        channel: int = 1,
        rng: Optional[random.Random] = None,
    ):
        self.bus = bus
        self.scales = scales
        self.name = name
        self.trigger_name = trigger_name
        self.scale_name = scale_name
        self.root_midi = root_midi
        self.pattern = pattern
        self.probability = probability
        self.gate_beats = gate_beats
        self.velocity = velocity
        self.channel = channel
        self.rng = rng or random.Random()
        self._i = 0

    def _next_pitch(self) -> int:
        scale = self.scales.get(self.scale_name)
        pat = self.pattern.lower()

        if pat == "root":
            return self.root_midi + scale.degree_to_semitones(0)
        elif pat == "octave":
            return (
                self.root_midi
                + scale.degree_to_semitones(0)
                + (12 if self._i % 2 == 0 else 0)
            )
        elif pat == "fifth":
            return self.root_midi + scale.degree_to_semitones(4)
        elif pat == "walk":
            degrees = [0, 0, 2, 4, 5, 7]
            d = self.rng.choice(degrees)
            return self.root_midi + scale.degree_to_semitones(d)
        else:
            return self.root_midi + scale.degree_to_semitones(0)

    async def start(self) -> None:
        trig_sub = await self.bus.subscribe(TriggerEvent)
        param_sub = await self.bus.subscribe(ParamChangeEvent)
        try:
            while True:
                done, _ = await asyncio.wait(
                    [
                        asyncio.create_task(trig_sub.recv()),
                        asyncio.create_task(param_sub.recv()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    ev = task.result()

                    if isinstance(ev, TriggerEvent):
                        if ev.name != self.trigger_name:
                            continue
                        if self.rng.random() > self.probability:
                            continue

                        pitch = self._next_pitch()
                        self._i += 1

                        await self.bus.publish(
                            NoteEvent(
                                pitch=pitch,
                                velocity=self.velocity,
                                gate_beats=self.gate_beats,
                                channel=self.channel,
                                meta={"source": self.name, "scale": self.scale_name},
                            )
                        )

                    elif isinstance(ev, ParamChangeEvent):
                        if ev.target != self.name:
                            continue
                        if ev.param == "scale":
                            self.scale_name = str(ev.value)
                        elif ev.param == "pattern":
                            self.pattern = str(ev.value)
                        elif ev.param == "probability":
                            self.probability = float(ev.value)
                        elif ev.param == "gate_beats":
                            self.gate_beats = float(ev.value)
                        elif ev.param == "velocity":
                            self.velocity = float(ev.value)
        finally:
            await trig_sub.close()
            await param_sub.close()
