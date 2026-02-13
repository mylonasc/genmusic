"""
Rhythm nodes: EuclideanRhythmNode, DrumNode.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..events import EventBus, TickEvent, TriggerEvent, ParamChangeEvent, NoteEvent
from .base import Node
from .timing import EuclideanPattern
from ..utils import DefaultRandomPolicy, RandomPolicy


class EuclideanRhythmNode(Node):
    """Generates trigger events based on Euclidean rhythm patterns."""

    def __init__(
        self,
        bus: EventBus,
        *,
        name: str = "euc",
        pattern: EuclideanPattern = None,
        hit_trigger_name: str = "rhythm_hit",
        probability: float = 1.0,
        rng: Optional[random.Random] = None,
        random_policy: RandomPolicy = None,
    ):
        self.bus = bus
        self.name = name
        self.pattern = pattern or EuclideanPattern()
        self.hit_trigger_name = hit_trigger_name
        self.probability = probability
        self.rng = rng or random.Random()
        self.random_policy = random_policy or DefaultRandomPolicy()
        self._rendered = self.pattern.render()

    def _rerender(self) -> None:
        self._rendered = self.pattern.render()

    async def start(self) -> None:
        tick_sub = await self.bus.subscribe(TickEvent)
        param_sub = await self.bus.subscribe(ParamChangeEvent)
        try:
            while True:
                done, _ = await asyncio.wait(
                    [
                        asyncio.create_task(tick_sub.recv()),
                        asyncio.create_task(param_sub.recv()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    ev = task.result()
                    if isinstance(ev, TickEvent):
                        if not self._rendered:
                            continue
                        idx = ev.step % len(self._rendered)
                        if self._rendered[idx] == 1 and self.random_policy.accept(
                            self.probability, rng=self.rng
                        ):
                            await self.bus.publish(
                                TriggerEvent(
                                    name=self.hit_trigger_name,
                                    value=1.0,
                                    meta={"source": self.name, "step": ev.step},
                                )
                            )
                    elif isinstance(ev, ParamChangeEvent):
                        if ev.target != self.name:
                            continue
                        if ev.param == "steps":
                            self.pattern = EuclideanPattern(
                                int(ev.value),
                                self.pattern.pulses,
                                self.pattern.rotation,
                            )
                            self._rerender()
                        elif ev.param == "pulses":
                            self.pattern = EuclideanPattern(
                                self.pattern.steps, int(ev.value), self.pattern.rotation
                            )
                            self._rerender()
                        elif ev.param == "rotation":
                            self.pattern = EuclideanPattern(
                                self.pattern.steps, self.pattern.pulses, int(ev.value)
                            )
                            self._rerender()
                        elif ev.param == "probability":
                            self.probability = float(ev.value)
        finally:
            await tick_sub.close()
            await param_sub.close()


@dataclass
class DrumConfig:
    """Configuration for drum patterns."""

    kick_pattern: Tuple[int, ...] = (1, 0, 0, 0, 1, 0, 0, 0)
    snare_pattern: Tuple[int, ...] = (0, 0, 0, 0, 1, 0, 0, 0)
    hihat_pattern: Tuple[int, ...] = (1, 0, 1, 0, 1, 0, 1, 0)
    kick_pitch: int = 36
    snare_pitch: int = 38
    hihat_pitch: int = 42
    velocity: float = 0.8
    channel: int = 9


class DrumNode(Node):
    """Generates drum sounds based on step patterns."""

    def __init__(
        self,
        bus: EventBus,
        *,
        name: str = "drums",
        trigger_name: str = "drum_trigger",
        config: DrumConfig = None,
    ):
        self.bus = bus
        self.name = name
        self.trigger_name = trigger_name
        self.config = config or DrumConfig()

    async def start(self) -> None:
        tick_sub = await self.bus.subscribe(TickEvent)
        try:
            while True:
                ev = await tick_sub.recv()
                step = ev.step % len(self.config.hihat_pattern)

                if self._trigger(step, self.config.kick_pattern):
                    await self.bus.publish(
                        NoteEvent(
                            pitch=self.config.kick_pitch,
                            velocity=self.config.velocity,
                            gate_beats=0.1,
                            channel=self.config.channel,
                            meta={"source": self.name, "drum": "kick"},
                        )
                    )

                if self._trigger(step, self.config.snare_pattern):
                    await self.bus.publish(
                        NoteEvent(
                            pitch=self.config.snare_pitch,
                            velocity=self.config.velocity,
                            gate_beats=0.1,
                            channel=self.config.channel,
                            meta={"source": self.name, "drum": "snare"},
                        )
                    )

                if self._trigger(step, self.config.hihat_pattern):
                    await self.bus.publish(
                        NoteEvent(
                            pitch=self.config.hihat_pitch,
                            velocity=self.config.velocity * 0.6,
                            gate_beats=0.05,
                            channel=self.config.channel,
                            meta={"source": self.name, "drum": "hihat"},
                        )
                    )
        finally:
            await tick_sub.close()

    def _trigger(self, step: int, pattern: Tuple[int, ...]) -> bool:
        if step >= len(pattern):
            return False
        return pattern[step] == 1
