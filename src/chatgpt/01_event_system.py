"""
Generative Algorithmic Music Control Framework (Python, asyncio, event-driven)

Key goals:
- Discrete + continuous control signals
- "Integration" of continuous signals into discrete events
- Multiple scales (arbitrary many)
- Arpeggiator whose behavior can change via events (e.g., change scale)
- Euclidean rhythm support
- Randomness policies controlling event consumption
- Non-blocking / asyncio-based

No actual audio output: the sink just prints NoteEvents by default.

Python 3.10+ recommended.
"""

from __future__ import annotations

import asyncio
import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from collections import defaultdict


# =========================
# Event model
# =========================

@dataclass(frozen=True)
class Event:
    """
    Base event. All events have a timestamp (monotonic seconds) and optional metadata.
    """
    t: float = field(default_factory=time.monotonic)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TickEvent(Event):
    """
    Emitted by a clock. 'step' is an integer index in a step-grid (e.g., 0..N-1 looping),
    'beat' is a monotonically increasing beat count (float or int).
    """
    step: int = 0
    beat: float = 0.0
    steps_per_beat: int = 4


@dataclass(frozen=True)
class TriggerEvent(Event):
    """
    Discrete impulse/trigger (e.g., rhythm hit, gate on).
    """
    name: str = "trigger"
    value: float = 1.0


@dataclass(frozen=True)
class ValueEvent(Event):
    """
    Continuous (or sampled) control value.
    """
    name: str = "value"
    value: float = 0.0


@dataclass(frozen=True)
class ParamChangeEvent(Event):
    """
    Generic parameter change. Use dotted paths if you want ("arp.pattern", "rhythm.k").
    """
    target: str = ""
    param: str = ""
    value: Any = None


@dataclass(frozen=True)
class NoteEvent(Event):
    """
    A musical note representation (no audio). 'pitch' can be MIDI note number or Hz,
    depending on your convention (here: default MIDI).
    """
    pitch: int = 60
    velocity: float = 0.8
    gate: float = 0.25  # in beats or seconds depending on downstream, here "beats-ish"
    channel: int = 0


# =========================
# Randomness policies
# =========================

class RandomPolicy(Protocol):
    """
    Controls probabilistic behavior: whether an event is consumed/acted upon, and
    optional random choice helper(s).
    """
    def accept(self, p: float, *, rng: random.Random) -> bool:
        ...

    def choice(self, seq: Sequence[Any], *, rng: random.Random) -> Any:
        ...


@dataclass
class DefaultRandomPolicy:
    def accept(self, p: float, *, rng: random.Random) -> bool:
        p = max(0.0, min(1.0, float(p)))
        return rng.random() < p

    def choice(self, seq: Sequence[Any], *, rng: random.Random) -> Any:
        return rng.choice(list(seq))


# =========================
# Event bus (pub/sub)
# =========================

E = TypeVar("E", bound=Event)

class EventBus:
    """
    Simple asyncio pub/sub bus.

    - Subscribers register for a specific Event type; they receive instances of that type
      (including subclasses).
    - Publishing is non-blocking: delivery uses asyncio queues.

    This keeps nodes decoupled. For heavier systems, you could add:
    - topic routing
    - backpressure policies
    - tracing/recording
    """

    def __init__(self) -> None:
        self._subs: DefaultDict[Type[Event], List[asyncio.Queue[Event]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, event_type: Type[E], max_queue: int = 1024) -> "Subscription[E]":
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue)
        async with self._lock:
            self._subs[event_type].append(q)
        return Subscription(bus=self, event_type=event_type, queue=q)

    async def unsubscribe(self, event_type: Type[Event], q: asyncio.Queue[Event]) -> None:
        async with self._lock:
            if event_type in self._subs and q in self._subs[event_type]:
                self._subs[event_type].remove(q)

    async def publish(self, ev: Event) -> None:
        # Deliver to subscribers registered for ev's type AND its base classes
        async with self._lock:
            targets: List[asyncio.Queue[Event]] = []
            for etype, qs in self._subs.items():
                if isinstance(ev, etype):
                    targets.extend(qs)

        # Non-blocking-ish delivery: if queues are full, drop (or you can block/log).
        for q in targets:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                # Drop policy: keep system running.
                pass


@dataclass
class Subscription(Protocol[E]):
    bus: EventBus
    event_type: Type[E]
    queue: asyncio.Queue[Event]

    async def recv(self) -> E:
        ev = await self.queue.get()
        return ev  # type: ignore[return-value]

    async def close(self) -> None:
        await self.bus.unsubscribe(self.event_type, self.queue)


@dataclass
class SubscriptionImpl:
    bus: EventBus
    event_type: Type[E]
    queue: asyncio.Queue[Event]

    async def recv(self) -> E:
        ev = await self.queue.get()
        return ev  # type: ignore[return-value]

    async def close(self) -> None:
        await self.bus.unsubscribe(self.event_type, self.queue)


# Monkey patch to keep typing light (so subscribe returns a concrete object)
EventBus.subscribe.__annotations__["return"] = SubscriptionImpl  # type: ignore


# =========================
# Scales
# =========================

@dataclass(frozen=True)
class Scale:
    """
    Scale defined by semitone intervals within an octave.
    Example major: [0,2,4,5,7,9,11]
    """
    name: str
    intervals: Tuple[int, ...]  # semitones from root within octave
    octave: int = 12

    def degree_to_semitones(self, degree: int) -> int:
        """
        Map an arbitrary degree (can exceed scale length) into semitone offset,
        wrapping through octaves.
        """
        n = len(self.intervals)
        if n == 0:
            return 0
        # allow negative degrees
        octave_shift, idx = divmod(degree, n)
        if degree < 0 and idx != 0:
            # Python's divmod for negative values needs careful interpretation
            # Example: divmod(-1, 7) -> (-1, 6) which is fine for our indexing,
            # but octave_shift already reflects that.
            pass
        return self.intervals[idx] + octave_shift * self.octave


class ScaleRegistry:
    def __init__(self) -> None:
        self._scales: Dict[str, Scale] = {}

    def add(self, scale: Scale) -> None:
        self._scales[scale.name] = scale

    def get(self, name: str) -> Scale:
        if name not in self._scales:
            raise KeyError(f"Unknown scale: {name}")
        return self._scales[name]

    def names(self) -> List[str]:
        return sorted(self._scales.keys())


# =========================
# Euclidean rhythm
# =========================

def bjorklund(k: int, n: int) -> List[int]:
    """
    Basic Bjorklund algorithm: distribute k pulses over n steps.
    Returns a list of 0/1 of length n.

    This implementation is compact and good enough for control-rate rhythms.
    """
    if n <= 0:
        return []
    k = max(0, min(k, n))
    if k == 0:
        return [0] * n
    if k == n:
        return [1] * n

    # Initialize
    pattern = []
    counts = []
    remainders = []
    divisor = n - k
    remainders.append(k)
    level = 0

    while True:
        counts.append(divisor // remainders[level])
        remainders.append(divisor % remainders[level])
        divisor = remainders[level]
        level += 1
        if remainders[level] <= 1:
            break
    counts.append(divisor)

    def build(level_: int) -> None:
        if level_ == -1:
            pattern.append(0)
        elif level_ == -2:
            pattern.append(1)
        else:
            for _ in range(counts[level_]):
                build(level_ - 1)
            if remainders[level_] != 0:
                build(level_ - 2)

    build(level)
    # rotate so pattern starts with a hit
    if 1 in pattern:
        first = pattern.index(1)
        pattern = pattern[first:] + pattern[:first]
    return pattern


@dataclass
class EuclideanPattern:
    steps: int
    pulses: int
    rotation: int = 0

    def render(self) -> List[int]:
        p = bjorklund(self.pulses, self.steps)
        if not p:
            return p
        r = self.rotation % len(p)
        return p[-r:] + p[:-r] if r else p


# =========================
# Core node protocol
# =========================

class Node(Protocol):
    """
    A running unit in the system. Start returns an asyncio Task (or tasks).
    """
    def start(self) -> Awaitable[None]:
        ...


# =========================
# Clock + Scheduler
# =========================

class Clock(Node):
    """
    Emits TickEvents at a fixed step rate derived from bpm and steps_per_beat.

    - Non-blocking: uses asyncio.sleep
    - For high-precision musical timing you’d eventually want a more careful scheduler
      (or audio callback timing), but this is fine for a control framework scaffold.
    """

    def __init__(
        self,
        bus: EventBus,
        bpm: float = 120.0,
        steps_per_beat: int = 4,
        loop_steps: int = 16,
        name: str = "clock",
    ) -> None:
        self.bus = bus
        self.bpm = bpm
        self.steps_per_beat = steps_per_beat
        self.loop_steps = loop_steps
        self.name = name
        self._running = False

    def seconds_per_step(self) -> float:
        beats_per_sec = self.bpm / 60.0
        steps_per_sec = beats_per_sec * self.steps_per_beat
        return 1.0 / max(1e-9, steps_per_sec)

    async def start(self) -> None:
        self._running = True
        step = 0
        beat = 0.0
        spstep = self.seconds_per_step()
        while self._running:
            await self.bus.publish(
                TickEvent(
                    step=step % self.loop_steps,
                    beat=beat,
                    steps_per_beat=self.steps_per_beat,
                    meta={"name": self.name},
                )
            )
            await asyncio.sleep(spstep)
            step += 1
            beat = step / self.steps_per_beat

    def stop(self) -> None:
        self._running = False


class EventScheduler(Node):
    """
    Schedules discrete events in the future (monotonic time seconds).
    Useful when something wants to emit events "later" without blocking.

    You can schedule from any node.
    """

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._heap: List[Tuple[float, int, Event]] = []
        self._counter = 0
        self._wakeup = asyncio.Event()
        self._running = False

    def schedule_in(self, delay_s: float, ev: Event) -> None:
        when = time.monotonic() + max(0.0, delay_s)
        self._counter += 1
        heapq.heappush(self._heap, (when, self._counter, ev))
        self._wakeup.set()

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

            # sleep until next event or earlier insertion
            self._wakeup.clear()
            timeout = max(0.0, when - now)
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._running = False
        self._wakeup.set()


# =========================
# Continuous signal sources + integrators
# =========================

class ContinuousSignal(Node):
    """
    Emits ValueEvent(name, value) periodically. The function can depend on time.
    """

    def __init__(
        self,
        bus: EventBus,
        name: str,
        fn: Callable[[float], float],
        hz: float = 50.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bus = bus
        self.name = name
        self.fn = fn
        self.hz = hz
        self.meta = meta or {}
        self._running = False

    async def start(self) -> None:
        self._running = True
        dt = 1.0 / max(1e-9, self.hz)
        t0 = time.monotonic()
        while self._running:
            t = time.monotonic() - t0
            v = float(self.fn(t))
            await self.bus.publish(ValueEvent(name=self.name, value=v, meta=self.meta))
            await asyncio.sleep(dt)

    def stop(self) -> None:
        self._running = False


class Integrator(Node):
    """
    Consumes ValueEvent(s) and integrates them over time. Can emit:
    - ValueEvent of the integrated state
    - TriggerEvent when thresholds crossed (useful for "integrating continuous to discrete")

    Example uses:
    - accumulate noise until it crosses 1.0 then trigger
    - integrate an LFO and trigger on wrap
    """

    def __init__(
        self,
        bus: EventBus,
        source_value_name: str,
        integrated_name: str,
        *,
        leak: float = 0.0,            # 0 => perfect integrator, >0 => leaky
        scale: float = 1.0,           # multiplier on incoming values
        emit_hz: float = 25.0,        # publish integrated value at this rate
        threshold: Optional[float] = None,
        threshold_trigger_name: str = "integrator_hit",
        rng: Optional[random.Random] = None,
        trigger_probability: float = 1.0,
        random_policy: RandomPolicy = DefaultRandomPolicy(),
    ) -> None:
        self.bus = bus
        self.source_value_name = source_value_name
        self.integrated_name = integrated_name
        self.leak = float(leak)
        self.scale = float(scale)
        self.emit_hz = float(emit_hz)
        self.threshold = threshold
        self.threshold_trigger_name = threshold_trigger_name
        self.rng = rng or random.Random()
        self.trigger_probability = float(trigger_probability)
        self.random_policy = random_policy

        self._running = False
        self._state = 0.0
        self._last_t = time.monotonic()

    async def start(self) -> None:
        sub = await self.bus.subscribe(ValueEvent)
        self._running = True

        # We'll also periodically emit the current integrated value
        emitter_task = asyncio.create_task(self._emitter_loop())

        try:
            while self._running:
                ev = await sub.recv()
                if ev.name != self.source_value_name:
                    continue

                now = time.monotonic()
                dt = max(0.0, now - self._last_t)
                self._last_t = now

                x = ev.value * self.scale
                # leaky integration: state' = x - leak*state
                # discrete update: state += (x - leak*state)*dt
                self._state += (x - self.leak * self._state) * dt

                if self.threshold is not None and self._state >= self.threshold:
                    if self.random_policy.accept(self.trigger_probability, rng=self.rng):
                        await self.bus.publish(
                            TriggerEvent(name=self.threshold_trigger_name, value=1.0, meta={"integrated": self._state})
                        )
                    # reset after hit (common pattern); you can make this configurable
                    self._state = 0.0
        finally:
            self._running = False
            emitter_task.cancel()
            await sub.close()

    async def _emitter_loop(self) -> None:
        dt = 1.0 / max(1e-9, self.emit_hz)
        while self._running:
            await self.bus.publish(ValueEvent(name=self.integrated_name, value=self._state))
            await asyncio.sleep(dt)

    def stop(self) -> None:
        self._running = False


# =========================
# Musical generators: Euclidean rhythm + Arpeggiator
# =========================

class EuclideanRhythmNode(Node):
    """
    On each TickEvent, checks its Euclidean pattern and emits TriggerEvent when hit.

    Supports runtime updates via ParamChangeEvent (steps/pulses/rotation, probability).
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        name: str = "euc",
        pattern: EuclideanPattern = EuclideanPattern(steps=16, pulses=5, rotation=0),
        hit_trigger_name: str = "rhythm_hit",
        probability: float = 1.0,
        rng: Optional[random.Random] = None,
        random_policy: RandomPolicy = DefaultRandomPolicy(),
    ) -> None:
        self.bus = bus
        self.name = name
        self.pattern = pattern
        self.hit_trigger_name = hit_trigger_name
        self.probability = float(probability)
        self.rng = rng or random.Random()
        self.random_policy = random_policy

        self._rendered = self.pattern.render()

    def _rerender(self) -> None:
        self._rendered = self.pattern.render()

    async def start(self) -> None:
        tick_sub = await self.bus.subscribe(TickEvent)
        param_sub = await self.bus.subscribe(ParamChangeEvent)
        try:
            while True:
                done, _ = await asyncio.wait(
                    [asyncio.create_task(tick_sub.recv()), asyncio.create_task(param_sub.recv())],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    ev = task.result()
                    if isinstance(ev, TickEvent):
                        if not self._rendered:
                            continue
                        idx = ev.step % len(self._rendered)
                        if self._rendered[idx] == 1 and self.random_policy.accept(self.probability, rng=self.rng):
                            await self.bus.publish(
                                TriggerEvent(name=self.hit_trigger_name, value=1.0, meta={"source": self.name, "step": ev.step})
                            )
                    elif isinstance(ev, ParamChangeEvent):
                        if ev.target != self.name:
                            continue
                        if ev.param == "steps":
                            self.pattern = EuclideanPattern(steps=int(ev.value), pulses=self.pattern.pulses, rotation=self.pattern.rotation)
                            self._rerender()
                        elif ev.param == "pulses":
                            self.pattern = EuclideanPattern(steps=self.pattern.steps, pulses=int(ev.value), rotation=self.pattern.rotation)
                            self._rerender()
                        elif ev.param == "rotation":
                            self.pattern = EuclideanPattern(steps=self.pattern.steps, pulses=self.pattern.pulses, rotation=int(ev.value))
                            self._rerender()
                        elif ev.param == "probability":
                            self.probability = float(ev.value)
        finally:
            await tick_sub.close()
            await param_sub.close()


@dataclass
class ArpConfig:
    """
    - degrees: scale degrees for the chord/pool (e.g. [0,2,4,7] for a 1-3-5-8)
    - pattern: 'up', 'down', 'updown', 'random'
    - octaves: how many octaves to span (>=1)
    """
    degrees: Tuple[int, ...] = (0, 2, 4, 7)
    pattern: str = "up"
    octaves: int = 1
    probability: float = 1.0
    gate: float = 0.25
    velocity: float = 0.8
    channel: int = 0


class Arpeggiator(Node):
    """
    Listens for TriggerEvent (e.g., rhythm hits) and emits NoteEvent according to:
    - current scale (from registry)
    - root note (MIDI)
    - ArpConfig pattern
    - changeable at runtime through ParamChangeEvent and custom events (e.g., change scale)

    You can route any trigger into it by matching trigger_name.
    """

    def __init__(
        self,
        bus: EventBus,
        scales: ScaleRegistry,
        *,
        name: str = "arp",
        trigger_name: str = "rhythm_hit",
        scale_name: str = "major",
        root_midi: int = 60,
        config: ArpConfig = ArpConfig(),
        rng: Optional[random.Random] = None,
        random_policy: RandomPolicy = DefaultRandomPolicy(),
    ) -> None:
        self.bus = bus
        self.scales = scales
        self.name = name
        self.trigger_name = trigger_name
        self.scale_name = scale_name
        self.root_midi = int(root_midi)
        self.config = config
        self.rng = rng or random.Random()
        self.random_policy = random_policy

        self._index = 0
        self._last_note: Optional[int] = None

    def _note_pool(self) -> List[int]:
        scale = self.scales.get(self.scale_name)
        pool: List[int] = []
        for octv in range(max(1, self.config.octaves)):
            base = octv * 12
            for d in self.config.degrees:
                semis = scale.degree_to_semitones(d) + base
                pool.append(self.root_midi + semis)
        # remove duplicates while preserving order
        seen = set()
        out = []
        for n in pool:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _next_note(self) -> int:
        pool = self._note_pool()
        if not pool:
            return self.root_midi

        pat = (self.config.pattern or "up").lower()
        if pat == "random":
            return self.random_policy.choice(pool, rng=self.rng)

        if pat == "down":
            idx = (-self._index - 1) % len(pool)
            self._index += 1
            return pool[idx]

        if pat == "updown":
            # bounce between ends
            cycle = pool + pool[-2:0:-1] if len(pool) > 1 else pool
            note = cycle[self._index % len(cycle)]
            self._index += 1
            return note

        # default "up"
        note = pool[self._index % len(pool)]
        self._index += 1
        return note

    async def start(self) -> None:
        trig_sub = await self.bus.subscribe(TriggerEvent)
        param_sub = await self.bus.subscribe(ParamChangeEvent)

        try:
            while True:
                done, _ = await asyncio.wait(
                    [asyncio.create_task(trig_sub.recv()), asyncio.create_task(param_sub.recv())],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    ev = task.result()

                    if isinstance(ev, TriggerEvent):
                        if ev.name != self.trigger_name:
                            continue
                        if not self.random_policy.accept(self.config.probability, rng=self.rng):
                            continue

                        pitch = self._next_note()
                        self._last_note = pitch
                        await self.bus.publish(
                            NoteEvent(
                                pitch=pitch,
                                velocity=float(self.config.velocity),
                                gate=float(self.config.gate),
                                channel=int(self.config.channel),
                                meta={"source": self.name, "scale": self.scale_name},
                            )
                        )

                    elif isinstance(ev, ParamChangeEvent):
                        if ev.target != self.name:
                            continue

                        # core params
                        if ev.param == "scale":
                            self.scale_name = str(ev.value)
                        elif ev.param == "root_midi":
                            self.root_midi = int(ev.value)

                        # arp config params
                        elif ev.param == "pattern":
                            self.config = ArpConfig(**{**self.config.__dict__, "pattern": str(ev.value)})
                        elif ev.param == "degrees":
                            degs = tuple(int(x) for x in ev.value)
                            self.config = ArpConfig(**{**self.config.__dict__, "degrees": degs})
                        elif ev.param == "octaves":
                            self.config = ArpConfig(**{**self.config.__dict__, "octaves": int(ev.value)})
                        elif ev.param == "probability":
                            self.config = ArpConfig(**{**self.config.__dict__, "probability": float(ev.value)})
                        elif ev.param == "gate":
                            self.config = ArpConfig(**{**self.config.__dict__, "gate": float(ev.value)})
                        elif ev.param == "velocity":
                            self.config = ArpConfig(**{**self.config.__dict__, "velocity": float(ev.value)})
                        elif ev.param == "channel":
                            self.config = ArpConfig(**{**self.config.__dict__, "channel": int(ev.value)})

        finally:
            await trig_sub.close()
            await param_sub.close()


# =========================
# Routing / Transform nodes
# =========================

class TriggerRouter(Node):
    """
    Re-maps or conditionally forwards triggers.
    Useful for:
    - mapping one rhythm output into multiple consumers
    - renaming triggers
    - probabilistic consumption / skipping
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        name: str = "router",
        in_name: str = "rhythm_hit",
        out_name: str = "rhythm_hit",
        probability: float = 1.0,
        rng: Optional[random.Random] = None,
        random_policy: RandomPolicy = DefaultRandomPolicy(),
        transform_value: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.bus = bus
        self.name = name
        self.in_name = in_name
        self.out_name = out_name
        self.probability = float(probability)
        self.rng = rng or random.Random()
        self.random_policy = random_policy
        self.transform_value = transform_value

    async def start(self) -> None:
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != self.in_name:
                    continue
                if not self.random_policy.accept(self.probability, rng=self.rng):
                    continue
                v = ev.value
                if self.transform_value:
                    v = float(self.transform_value(v))
                await self.bus.publish(TriggerEvent(name=self.out_name, value=v, meta={**ev.meta, "routed_by": self.name}))
        finally:
            await sub.close()


# =========================
# Sinks / Debug
# =========================

class PrintNoteSink(Node):
    """
    Prints NoteEvents. Replace with MIDI out / OSC / your synth engine later.
    """

    def __init__(self, bus: EventBus, *, name: str = "print_sink") -> None:
        self.bus = bus
        self.name = name

    async def start(self) -> None:
        sub = await self.bus.subscribe(NoteEvent)
        try:
            while True:
                ev = await sub.recv()
                print(
                    f"[{self.name}] Note pitch={ev.pitch} vel={ev.velocity:.2f} gate={ev.gate:.2f} "
                    f"ch={ev.channel} meta={ev.meta}"
                )
        finally:
            await sub.close()


# =========================
# System container
# =========================

class System:
    """
    Simple container to start/stop multiple nodes.
    """

    def __init__(self) -> None:
        self.bus = EventBus()
        self.nodes: List[Node] = []
        self._tasks: List[asyncio.Task[None]] = []

    def add(self, node: Node) -> None:
        self.nodes.append(node)

    async def run(self) -> None:
        # Start all nodes concurrently
        for n in self.nodes:
            self._tasks.append(asyncio.create_task(n.start()))
        await asyncio.gather(*self._tasks)

    async def run_for(self, seconds: float) -> None:
        for n in self.nodes:
            self._tasks.append(asyncio.create_task(n.start()))
        try:
            await asyncio.sleep(max(0.0, seconds))
        finally:
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)


# =========================
# Example wiring (optional)
# =========================

def default_scales() -> ScaleRegistry:
    reg = ScaleRegistry()
    reg.add(Scale("major", (0, 2, 4, 5, 7, 9, 11)))
    reg.add(Scale("minor", (0, 2, 3, 5, 7, 8, 10)))
    reg.add(Scale("dorian", (0, 2, 3, 5, 7, 9, 10)))
    reg.add(Scale("phrygian", (0, 1, 3, 5, 7, 8, 10)))
    reg.add(Scale("lydian", (0, 2, 4, 6, 7, 9, 11)))
    reg.add(Scale("mixolydian", (0, 2, 4, 5, 7, 9, 10)))
    reg.add(Scale("locrian", (0, 1, 3, 5, 6, 8, 10)))
    reg.add(Scale("chromatic", tuple(range(12))))
    return reg


async def demo() -> None:
    """
    Minimal demo:
    - clock (16-step loop)
    - euclidean rhythm produces hits
    - arpeggiator consumes hits and emits notes
    - continuous noise integrates into occasional triggers that change scale

    This demonstrates:
    - discrete event flow (Tick -> Trigger -> Note)
    - continuous -> integration -> discrete trigger -> param change
    """
    sys = System()
    bus = sys.bus

    scales = default_scales()

    clock = Clock(bus, bpm=120, steps_per_beat=4, loop_steps=16)
    rhythm = EuclideanRhythmNode(
        bus,
        name="euc",
        pattern=EuclideanPattern(steps=16, pulses=5, rotation=0),
        hit_trigger_name="rhythm_hit",
        probability=0.95,
    )
    arp = Arpeggiator(
        bus,
        scales,
        name="arp",
        trigger_name="rhythm_hit",
        scale_name="dorian",
        root_midi=60,
        config=ArpConfig(degrees=(0, 2, 4, 6), pattern="updown", octaves=2, probability=1.0, gate=0.20),
    )
    sink = PrintNoteSink(bus)

    # Continuous signal: random-ish (smooth noise) using a cheap sum of sines
    lfo = ContinuousSignal(
        bus,
        name="mod_noise",
        hz=50.0,
        fn=lambda t: 0.45 * math.sin(2.0 * math.pi * 0.37 * t)
                    + 0.35 * math.sin(2.0 * math.pi * 0.71 * t + 1.0)
                    + 0.20 * math.sin(2.0 * math.pi * 1.13 * t + 2.2),
    )

    # Integrate continuous signal; when threshold crossed, fire trigger
    integ = Integrator(
        bus,
        source_value_name="mod_noise",
        integrated_name="mod_noise_int",
        leak=0.05,
        scale=1.0,
        threshold=0.8,
        threshold_trigger_name="scale_flip",
        trigger_probability=0.8,
    )

    # A small node that changes the arp scale when scale_flip triggers
    class ScaleFlipNode(Node):
        def __init__(self, bus: EventBus) -> None:
            self.bus = bus
            self._names = ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian"]
            self._i = 0

        async def start(self) -> None:
            sub = await self.bus.subscribe(TriggerEvent)
            try:
                while True:
                    ev = await sub.recv()
                    if ev.name != "scale_flip":
                        continue
                    self._i = (self._i + 1) % len(self._names)
                    await self.bus.publish(ParamChangeEvent(target="arp", param="scale", value=self._names[self._i]))
            finally:
                await sub.close()

    sys.add(clock)
    sys.add(rhythm)
    sys.add(arp)
    sys.add(sink)
    sys.add(lfo)
    sys.add(integ)
    sys.add(ScaleFlipNode(bus))

    # run for a short time
    await sys.run_for(8.0)


if __name__ == "__main__":
    asyncio.run(demo())
