"""
Arpeggiator node.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

from ..events import EventBus, TriggerEvent, ParamChangeEvent, NoteEvent
from ..scales import ScaleRegistry
from .base import Node
from ..utils import DefaultRandomPolicy, RandomPolicy


@dataclass
class ArpConfig:
    """Configuration for arpeggiator."""

    degrees: Tuple[int, ...] = (0, 2, 4, 7)
    pattern: str = "updown"
    octaves: int = 2
    probability: float = 1.0
    gate_beats: float = 0.25
    velocity: float = 0.8
    channel: int = 0


class Arpeggiator(Node):
    """Generates melodic notes from triggers based on scales."""

    def __init__(
        self,
        bus: EventBus,
        scales: ScaleRegistry,
        *,
        name: str = "arp",
        trigger_name: str = "rhythm_hit",
        scale_name: str = "dorian",
        root_midi: int = 60,
        config: ArpConfig = None,
        rng: Optional[random.Random] = None,
        random_policy: RandomPolicy = None,
    ):
        self.bus = bus
        self.scales = scales
        self.name = name
        self.trigger_name = trigger_name
        self.scale_name = scale_name
        self.root_midi = root_midi
        self.config = config or ArpConfig()
        self.rng = rng or random.Random()
        self.random_policy = random_policy or DefaultRandomPolicy()
        self._i = 0

    def _pool(self) -> List[int]:
        scale = self.scales.get(self.scale_name)
        out: List[int] = []
        for o in range(max(1, self.config.octaves)):
            base = o * 12
            for d in self.config.degrees:
                out.append(self.root_midi + scale.degree_to_semitones(d) + base)
        seen, uniq = set(), []
        for n in out:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        return uniq

    def _next(self) -> int:
        pool = self._pool()
        if not pool:
            return self.root_midi
        pat = self.config.pattern.lower()
        if pat == "random":
            return self.random_policy.choice(pool, rng=self.rng)
        if pat == "down":
            idx = (-self._i - 1) % len(pool)
            self._i += 1
            return pool[idx]
        if pat == "updown":
            cycle = pool + pool[-2:0:-1] if len(pool) > 1 else pool
            n = cycle[self._i % len(cycle)]
            self._i += 1
            return n
        n = pool[self._i % len(pool)]
        self._i += 1
        return n

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
                        if not self.random_policy.accept(
                            self.config.probability, rng=self.rng
                        ):
                            continue
                        pitch = self._next()
                        await self.bus.publish(
                            NoteEvent(
                                pitch=pitch,
                                velocity=self.config.velocity,
                                gate_beats=self.config.gate_beats,
                                channel=self.config.channel,
                                meta={"source": self.name, "scale": self.scale_name},
                            )
                        )
                    elif isinstance(ev, ParamChangeEvent):
                        if ev.target != self.name:
                            continue
                        if ev.param == "scale":
                            self.scale_name = str(ev.value)
                        elif ev.param == "root_midi":
                            self.root_midi = int(ev.value)
                        elif ev.param == "pattern":
                            self.config = ArpConfig(
                                **{**self.config.__dict__, "pattern": str(ev.value)}
                            )
                        elif ev.param == "degrees":
                            self.config = ArpConfig(
                                **{
                                    **self.config.__dict__,
                                    "degrees": tuple(int(x) for x in ev.value),
                                }
                            )
                        elif ev.param == "octaves":
                            self.config = ArpConfig(
                                **{**self.config.__dict__, "octaves": int(ev.value)}
                            )
                        elif ev.param == "gate_beats":
                            self.config = ArpConfig(
                                **{
                                    **self.config.__dict__,
                                    "gate_beats": float(ev.value),
                                }
                            )
                        elif ev.param == "velocity":
                            self.config = ArpConfig(
                                **{**self.config.__dict__, "velocity": float(ev.value)}
                            )
                        elif ev.param == "probability":
                            self.config = ArpConfig(
                                **{
                                    **self.config.__dict__,
                                    "probability": float(ev.value),
                                }
                            )
        finally:
            await trig_sub.close()
            await param_sub.close()
