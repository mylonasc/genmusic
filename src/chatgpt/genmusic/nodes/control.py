"""
Control nodes: ContinuousSignal, Integrator.
"""

import asyncio
import random
import time
from typing import Callable, Optional

from ..events import EventBus, ValueEvent, TriggerEvent
from .base import Node
from ..utils import DefaultRandomPolicy, RandomPolicy


class ContinuousSignal(Node):
    """Generates continuous value updates from a function."""

    def __init__(
        self,
        bus: EventBus,
        name: str,
        fn: Callable[[float], float],
        hz: float = 50.0,
    ):
        self.bus = bus
        self.name = name
        self.fn = fn
        self.hz = hz
        self._running = False

    async def start(self) -> None:
        self._running = True
        dt = 1.0 / max(1e-9, self.hz)
        t0 = time.monotonic()
        while self._running:
            t = time.monotonic() - t0
            await self.bus.publish(ValueEvent(name=self.name, value=float(self.fn(t))))
            await asyncio.sleep(dt)

    def stop(self) -> None:
        self._running = False


class Integrator(Node):
    """Integrates continuous values and triggers when threshold is crossed."""

    def __init__(
        self,
        bus: EventBus,
        source_value_name: str,
        *,
        leak: float = 0.0,
        scale: float = 1.0,
        threshold: Optional[float] = None,
        trigger_name: str = "integrator_hit",
        trigger_probability: float = 1.0,
        rng: Optional[random.Random] = None,
        random_policy: RandomPolicy = None,
    ):
        self.bus = bus
        self.source_value_name = source_value_name
        self.leak = leak
        self.scale = scale
        self.threshold = threshold
        self.trigger_name = trigger_name
        self.trigger_probability = trigger_probability
        self.rng = rng or random.Random()
        self.random_policy = random_policy or DefaultRandomPolicy()
        self._state = 0.0
        self._last_t = time.monotonic()

    async def start(self) -> None:
        sub = await self.bus.subscribe(ValueEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != self.source_value_name:
                    continue
                now = time.monotonic()
                dt = max(0.0, now - self._last_t)
                self._last_t = now

                x = ev.value * self.scale
                self._state += (x - self.leak * self._state) * dt

                if self.threshold is not None and self._state >= self.threshold:
                    if self.random_policy.accept(
                        self.trigger_probability, rng=self.rng
                    ):
                        await self.bus.publish(
                            TriggerEvent(
                                name=self.trigger_name,
                                value=1.0,
                                meta={"integrated": self._state},
                            )
                        )
                    self._state = 0.0
        finally:
            await sub.close()
