"""
Event system for genmusic.

Provides:
- Base Event classes (Event, TickEvent, TriggerEvent, ValueEvent, ParamChangeEvent, NoteEvent)
- EventBus: async pub/sub with support for event filtering, chaining, and delayed scheduling
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
)
from collections import defaultdict
import heapq


E = TypeVar("E", bound="Event")
T = TypeVar("T")


@dataclass(frozen=True)
class Event:
    """Base event class. All events have a timestamp and metadata."""

    t: float = field(default_factory=time.monotonic)
    meta: Dict[str, Any] = field(default_factory=dict)

    def with_meta(self, **kwargs) -> "Event":
        """Return a copy with updated metadata."""
        new_meta = {**self.meta, **kwargs}
        return Event(t=self.t, meta=new_meta)


@dataclass(frozen=True)
class TickEvent(Event):
    """Periodic timing event from the clock."""

    step: int = 0
    beat: float = 0.0
    steps_per_beat: int = 4


@dataclass(frozen=True)
class TriggerEvent(Event):
    """Discrete trigger/impulse event."""

    name: str = "trigger"
    value: float = 1.0


@dataclass(frozen=True)
class ValueEvent(Event):
    """Continuous value update event."""

    name: str = "value"
    value: float = 0.0


@dataclass(frozen=True)
class ParamChangeEvent(Event):
    """Parameter change event for runtime control."""

    target: str = ""
    param: str = ""
    value: Any = None


@dataclass(frozen=True)
class NoteEvent(Event):
    """Musical note event."""

    pitch: int = 60
    velocity: float = 0.8
    gate_beats: float = 0.25
    channel: int = 0


class EventFilter:
    """Filter for events based on type and/or attributes."""

    def __init__(
        self,
        event_type: Optional[Type[Event]] = None,
        name: Optional[str] = None,
        target: Optional[str] = None,
        channel: Optional[int] = None,
        predicate: Optional[Callable[[Event], bool]] = None,
    ):
        self.event_type = event_type
        self.name = name
        self.target = target
        self.channel = channel
        self.predicate = predicate

    def matches(self, ev: Event) -> bool:
        if self.event_type and not isinstance(ev, self.event_type):
            return False
        if self.name and isinstance(ev, TriggerEvent):
            if ev.name != self.name:
                return False
        if self.target and isinstance(ev, ParamChangeEvent):
            if ev.target != self.target:
                return False
        if self.channel is not None and isinstance(ev, NoteEvent):
            if ev.channel != self.channel:
                return False
        if self.predicate and not self.predicate(ev):
            return False
        return True


class Subscription:
    """Handle for managing event subscriptions."""

    def __init__(
        self,
        bus: "EventBus",
        event_type: Type[E],
        queue: asyncio.Queue[Event],
        filter: Optional[EventFilter] = None,
    ):
        self.bus = bus
        self.event_type = event_type
        self.queue = queue
        self.filter = filter
        self._closed = False

    async def recv(self) -> E:
        """Receive the next matching event."""
        while True:
            ev = await self.queue.get()
            if self.filter is None or self.filter.matches(ev):
                return ev  # type: ignore

    def try_recv(self) -> Optional[E]:
        """Non-blocking receive. Returns None if no event available."""
        try:
            ev = self.queue.get_nowait()
            if self.filter is None or self.filter.matches(ev):
                return ev
        except asyncio.QueueEmpty:
            pass
        return None

    async def close(self) -> None:
        """Unsubscribe from events."""
        if self._closed:
            return
        self._closed = True
        await self.bus._unsubscribe(self.event_type, self.queue)


class EventChain:
    """Event chain - transforms/triggers new events based on incoming events."""

    def __init__(
        self,
        bus: EventBus,
        source_filter: EventFilter,
        transform: Callable[[Event], Optional[Event]],
    ):
        self.bus = bus
        self.source_filter = source_filter
        self.transform = transform
        self._subscription: Optional[Subscription] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start processing the chain."""
        self._subscription = await self.bus.subscribe(
            self.source_filter.event_type, filter=self.source_filter
        )
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        if not self._subscription:
            return
        try:
            while True:
                ev = await self._subscription.recv()
                new_ev = self.transform(ev)
                if new_ev:
                    await self.bus.publish(new_ev)
        except asyncio.CancelledError:
            pass
        finally:
            if self._subscription:
                await self._subscription.close()

    def stop(self) -> None:
        """Stop the chain."""
        if self._task:
            self._task.cancel()


class EventBus:
    """
    Async event bus with pub/sub, filtering, chaining, and delayed scheduling.

    Features:
    - Subscribe to events by type with optional filters
    - Event chains: transform/trigger new events from incoming events
    - Delayed publishing: schedule events for future delivery
    - Event transformation pipelines
    """

    def __init__(self):
        self._subs: Dict[Type[Event], List[asyncio.Queue[Event]]] = defaultdict(list)
        self._filters: Dict[asyncio.Queue[Event], EventFilter] = {}
        self._lock = asyncio.Lock()
        self._chains: List[EventChain] = []

    async def subscribe(
        self,
        event_type: Type[E],
        filter: Optional[EventFilter] = None,
        max_queue: int = 2048,
    ) -> Subscription:
        """Subscribe to events of a specific type, optionally filtered."""
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue)
        async with self._lock:
            self._subs[event_type].append(queue)
            if filter:
                self._filters[queue] = filter
        return Subscription(bus=self, event_type=event_type, queue=queue, filter=filter)

    async def _unsubscribe(
        self, event_type: Type[Event], queue: asyncio.Queue[Event]
    ) -> None:
        async with self._lock:
            if event_type in self._subs and queue in self._subs[event_type]:
                self._subs[event_type].remove(queue)
            if queue in self._filters:
                del self._filters[queue]

    async def publish(self, ev: Event) -> None:
        """Publish an event to all subscribers."""
        async with self._lock:
            targets: List[asyncio.Queue[Event]] = []
            for etype, queues in self._subs.items():
                if isinstance(ev, etype):
                    targets.extend(queues)

        for q in targets:
            filt = self._filters.get(q)
            if filt is None or filt.matches(ev):
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    pass

    async def publish_delayed(self, ev: Event, delay: float) -> None:
        """Publish an event after a delay."""
        await asyncio.sleep(delay)
        await self.publish(ev)

    def publish_delayed_bg(self, ev: Event, delay: float) -> "DelayedEventHandle":
        """Publish an event after a delay (background, returns handle)."""
        return DelayedEventHandle(self, ev, delay)

    def create_chain(
        self,
        source: EventFilter,
        transform: Callable[[Event], Optional[Event]],
    ) -> EventChain:
        """Create an event chain that transforms/triggers new events."""
        chain = EventChain(self, source, transform)
        self._chains.append(chain)
        return chain

    async def start_chains(self) -> None:
        """Start all registered event chains."""
        for chain in self._chains:
            await chain.start()

    def stop_chains(self) -> None:
        """Stop all event chains."""
        for chain in self._chains:
            chain.stop()


class DelayedEventHandle:
    """Handle for managing a delayed event publication."""

    def __init__(self, bus: EventBus, ev: Event, delay: float):
        self.bus = bus
        self.ev = ev
        self.delay = delay
        self._cancelled = False
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start the delayed publication."""

        async def run():
            if self._cancelled:
                return
            await asyncio.sleep(self.delay)
            if not self._cancelled:
                await self.bus.publish(self.ev)

        self._task = asyncio.create_task(run())

    def cancel(self) -> None:
        """Cancel the delayed publication."""
        self._cancelled = True
        if self._task:
            self._task.cancel()


# Backwards compatibility alias
SubscriptionImpl = Subscription
