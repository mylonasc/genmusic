"""
Timing nodes: Clock and EventScheduler.
"""

import asyncio
import heapq
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..events import Event, EventBus, TickEvent, TriggerEvent
from .base import Node


@dataclass
class EuclideanPattern:
    """Euclidean rhythm pattern configuration."""

    steps: int = 16
    pulses: int = 5
    rotation: int = 0

    def render(self) -> List[int]:
        from ..utils import euclidean_pattern

        return euclidean_pattern(self.steps, self.pulses, self.rotation)


class Clock(Node):
    """Clock that generates periodic TickEvents."""

    def __init__(
        self,
        bus: EventBus,
        bpm: float = 120.0,
        steps_per_beat: int = 4,
        loop_steps: int = 16,
    ):
        self.bus = bus
        self.bpm = bpm
        self.steps_per_beat = steps_per_beat
        self.loop_steps = loop_steps
        self._running = False

    def seconds_per_step(self) -> float:
        beats_per_sec = self.bpm / 60.0
        steps_per_sec = beats_per_sec * self.steps_per_beat
        return 1.0 / max(1e-9, steps_per_sec)

    async def start(self) -> None:
        self._running = True
        step = 0
        spstep = self.seconds_per_step()
        while self._running:
            await self.bus.publish(
                TickEvent(
                    step=step % self.loop_steps,
                    beat=step / self.steps_per_beat,
                    steps_per_beat=self.steps_per_beat,
                )
            )
            await asyncio.sleep(spstep)
            step += 1

    def stop(self) -> None:
        self._running = False


class EventScheduler(Node):
    """Scheduler for timed event delivery."""

    def __init__(self, bus: EventBus):
        self.bus = bus
        self._heap: List[Tuple[float, int, Event]] = []
        self._counter = 0
        self._wakeup = asyncio.Event()
        self._running = False

    def schedule_at(self, when_monotonic: float, ev: Event) -> None:
        """Schedule an event at a specific time."""
        self._counter += 1
        heapq.heappush(self._heap, (when_monotonic, self._counter, ev))
        self._wakeup.set()

    def schedule_in(self, delay_s: float, ev: Event) -> None:
        """Schedule an event after a delay."""
        self.schedule_at(time.monotonic() + max(0.0, delay_s), ev)

    async def start(self) -> None:
        self._running = True
        while self._running:
            now = time.monotonic()
            if not self._heap:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            when, _, ev = self._heap[0]
            if when <= now:
                heapq.heappop(self._heap)
                await self.bus.publish(ev)
                continue
            self._wakeup.clear()
            try:
                await asyncio.wait_for(
                    self._wakeup.wait(), timeout=max(0.0, when - now)
                )
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._running = False
        self._wakeup.set()
