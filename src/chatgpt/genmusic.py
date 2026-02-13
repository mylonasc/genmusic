"""
Generative Algorithmic Music Control Framework + MIDI output (asyncio)

How to hear sound:
- Install: pip install mido python-rtmidi
- Run: python this_file.py
- If a MIDI output port exists, it will stream notes live.
- Otherwise it writes out.mid (looped phrases) which you can open in a DAW/player.

Optional:
- On macOS you can route to "IAC Driver" bus and use a synth.
- On Windows use e.g. loopMIDI + a softsynth.
- On Linux use a2jmidid + fluidsynth or a DAW.
"""

from __future__ import annotations

import asyncio
import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Protocol, Sequence, Tuple, Type, TypeVar
from collections import defaultdict

# -------- MIDI deps (optional runtime) --------
try:
    import mido
except Exception:
    mido = None  # type: ignore


# =========================
# Event model
# =========================

@dataclass(frozen=True)
class Event:
    t: float = field(default_factory=time.monotonic)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TickEvent(Event):
    step: int = 0
    beat: float = 0.0
    steps_per_beat: int = 4

@dataclass(frozen=True)
class TriggerEvent(Event):
    name: str = "trigger"
    value: float = 1.0

@dataclass(frozen=True)
class ValueEvent(Event):
    name: str = "value"
    value: float = 0.0

@dataclass(frozen=True)
class ParamChangeEvent(Event):
    target: str = ""
    param: str = ""
    value: Any = None

@dataclass(frozen=True)
class NoteEvent(Event):
    pitch: int = 60
    velocity: float = 0.8
    gate_beats: float = 0.25  # duration in beats
    channel: int = 0


# =========================
# Randomness policies
# =========================

class RandomPolicy(Protocol):
    def accept(self, p: float, *, rng: random.Random) -> bool: ...
    def choice(self, seq: Sequence[Any], *, rng: random.Random) -> Any: ...

@dataclass
class DefaultRandomPolicy:
    def accept(self, p: float, *, rng: random.Random) -> bool:
        p = max(0.0, min(1.0, float(p)))
        return rng.random() < p
    def choice(self, seq: Sequence[Any], *, rng: random.Random) -> Any:
        return rng.choice(list(seq))


# =========================
# Event bus
# =========================

E = TypeVar("E", bound=Event)

class EventBus:
    def __init__(self) -> None:
        self._subs: DefaultDict[Type[Event], List[asyncio.Queue[Event]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, event_type: Type[E], max_queue: int = 2048) -> "SubscriptionImpl[E]":
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue)
        async with self._lock:
            self._subs[event_type].append(q)
        return SubscriptionImpl(bus=self, event_type=event_type, queue=q)

    async def unsubscribe(self, event_type: Type[Event], q: asyncio.Queue[Event]) -> None:
        async with self._lock:
            if event_type in self._subs and q in self._subs[event_type]:
                self._subs[event_type].remove(q)

    async def publish(self, ev: Event) -> None:
        async with self._lock:
            targets: List[asyncio.Queue[Event]] = []
            for etype, qs in self._subs.items():
                if isinstance(ev, etype):
                    targets.extend(qs)
        for q in targets:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass

@dataclass
class SubscriptionImpl:
    bus: EventBus
    event_type: Type[E]
    queue: asyncio.Queue[Event]
    async def recv(self) -> E:
        ev = await self.queue.get()
        return ev  # type: ignore
    async def close(self) -> None:
        await self.bus.unsubscribe(self.event_type, self.queue)


# =========================
# Scales
# =========================

@dataclass(frozen=True)
class Scale:
    name: str
    intervals: Tuple[int, ...]
    octave: int = 12

    def degree_to_semitones(self, degree: int) -> int:
        n = len(self.intervals)
        if n == 0:
            return 0
        octave_shift, idx = divmod(degree, n)
        return self.intervals[idx] + octave_shift * self.octave

class ScaleRegistry:
    def __init__(self) -> None:
        self._scales: Dict[str, Scale] = {}
    def add(self, scale: Scale) -> None:
        self._scales[scale.name] = scale
    def get(self, name: str) -> Scale:
        return self._scales[name]
    def names(self) -> List[str]:
        return sorted(self._scales.keys())


# =========================
# Euclidean rhythm
# =========================

def bjorklund(k: int, n: int) -> List[int]:
    if n <= 0:
        return []
    k = max(0, min(k, n))
    if k == 0:
        return [0] * n
    if k == n:
        return [1] * n

    pattern, counts, remainders = [], [], []
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
    if 1 in pattern:
        first = pattern.index(1)
        pattern[:] = pattern[first:] + pattern[:first]
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
# Nodes
# =========================

class Node(Protocol):
    async def start(self) -> None: ...


class Clock(Node):
    def __init__(self, bus: EventBus, bpm: float = 120.0, steps_per_beat: int = 4, loop_steps: int = 16) -> None:
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
            await self.bus.publish(TickEvent(step=step % self.loop_steps, beat=step / self.steps_per_beat, steps_per_beat=self.steps_per_beat))
            await asyncio.sleep(spstep)
            step += 1

    def stop(self) -> None:
        self._running = False


class EventScheduler(Node):
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._heap: List[Tuple[float, int, Event]] = []
        self._counter = 0
        self._wakeup = asyncio.Event()
        self._running = False

    def schedule_at(self, when_monotonic: float, ev: Event) -> None:
        self._counter += 1
        heapq.heappush(self._heap, (when_monotonic, self._counter, ev))
        self._wakeup.set()

    def schedule_in(self, delay_s: float, ev: Event) -> None:
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
                await asyncio.wait_for(self._wakeup.wait(), timeout=max(0.0, when - now))
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._running = False
        self._wakeup.set()


class ContinuousSignal(Node):
    def __init__(self, bus: EventBus, name: str, fn: Callable[[float], float], hz: float = 50.0) -> None:
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
        random_policy: RandomPolicy = DefaultRandomPolicy(),
    ) -> None:
        self.bus = bus
        self.source_value_name = source_value_name
        self.leak = leak
        self.scale = scale
        self.threshold = threshold
        self.trigger_name = trigger_name
        self.trigger_probability = trigger_probability
        self.rng = rng or random.Random()
        self.random_policy = random_policy
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
                    if self.random_policy.accept(self.trigger_probability, rng=self.rng):
                        await self.bus.publish(TriggerEvent(name=self.trigger_name, value=1.0, meta={"integrated": self._state}))
                    self._state = 0.0
        finally:
            await sub.close()


class EuclideanRhythmNode(Node):
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
        self.probability = probability
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
                            await self.bus.publish(TriggerEvent(name=self.hit_trigger_name, value=1.0, meta={"source": self.name, "step": ev.step}))
                    elif isinstance(ev, ParamChangeEvent):
                        if ev.target != self.name:
                            continue
                        if ev.param == "steps":
                            self.pattern = EuclideanPattern(int(ev.value), self.pattern.pulses, self.pattern.rotation)
                            self._rerender()
                        elif ev.param == "pulses":
                            self.pattern = EuclideanPattern(self.pattern.steps, int(ev.value), self.pattern.rotation)
                            self._rerender()
                        elif ev.param == "rotation":
                            self.pattern = EuclideanPattern(self.pattern.steps, self.pattern.pulses, int(ev.value))
                            self._rerender()
                        elif ev.param == "probability":
                            self.probability = float(ev.value)
        finally:
            await tick_sub.close()
            await param_sub.close()


@dataclass
class ArpConfig:
    degrees: Tuple[int, ...] = (0, 2, 4, 7)
    pattern: str = "updown"
    octaves: int = 2
    probability: float = 1.0
    gate_beats: float = 0.25
    velocity: float = 0.8
    channel: int = 0


class Arpeggiator(Node):
    def __init__(
        self,
        bus: EventBus,
        scales: ScaleRegistry,
        *,
        name: str = "arp",
        trigger_name: str = "rhythm_hit",
        scale_name: str = "dorian",
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
        self.root_midi = root_midi
        self.config = config
        self.rng = rng or random.Random()
        self.random_policy = random_policy
        self._i = 0

    def _pool(self) -> List[int]:
        scale = self.scales.get(self.scale_name)
        out: List[int] = []
        for o in range(max(1, self.config.octaves)):
            base = o * 12
            for d in self.config.degrees:
                out.append(self.root_midi + scale.degree_to_semitones(d) + base)
        # uniq
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
        # up
        n = pool[self._i % len(pool)]
        self._i += 1
        return n

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
                            self.config = ArpConfig(**{**self.config.__dict__, "pattern": str(ev.value)})
                        elif ev.param == "degrees":
                            self.config = ArpConfig(**{**self.config.__dict__, "degrees": tuple(int(x) for x in ev.value)})
                        elif ev.param == "octaves":
                            self.config = ArpConfig(**{**self.config.__dict__, "octaves": int(ev.value)})
                        elif ev.param == "gate_beats":
                            self.config = ArpConfig(**{**self.config.__dict__, "gate_beats": float(ev.value)})
                        elif ev.param == "velocity":
                            self.config = ArpConfig(**{**self.config.__dict__, "velocity": float(ev.value)})
                        elif ev.param == "probability":
                            self.config = ArpConfig(**{**self.config.__dict__, "probability": float(ev.value)})
        finally:
            await trig_sub.close()
            await param_sub.close()


# =========================
# MIDI output sink
# =========================

class MidiOutSink(Node):
    """
    Converts NoteEvent -> MIDI note_on/note_off
    - If an output port is available, sends live.
    - Always records to a MidiFile in memory; on shutdown writes out.mid.

    Timing:
    - note durations use beat->seconds conversion using provided bpm.
    """

    def __init__(
        self,
        bus: EventBus,
        scheduler: EventScheduler,
        *,
        bpm: float,
        out_path: str = "out.mid",
        preferred_port_contains: Optional[str] = None,
    ) -> None:
        self.bus = bus
        self.scheduler = scheduler
        self.bpm = bpm
        self.out_path = out_path
        self.preferred_port_contains = preferred_port_contains

        self._port = None
        self._mid = None
        self._track = None
        self._last_write_time = time.monotonic()

    def _beats_to_seconds(self, beats: float) -> float:
        return (60.0 / max(1e-9, self.bpm)) * float(beats)

    def _open_midi(self) -> None:
        if mido is None:
            print("[MIDI] mido not installed. Install with: pip install mido python-rtmidi")
            return

        # Prepare file recording
        self._mid = mido.MidiFile(ticks_per_beat=480)
        self._track = mido.MidiTrack()
        self._mid.tracks.append(self._track)
        self._track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(self.bpm), time=0))

        # Try open output port
        try:
            names = mido.get_output_names()
            if names:
                chosen = None
                if self.preferred_port_contains:
                    for n in names:
                        if self.preferred_port_contains.lower() in n.lower():
                            chosen = n
                            break
                chosen = chosen or names[0]
                self._port = mido.open_output(chosen)
                print(f"[MIDI] Live output enabled on: {chosen}")
            else:
                print("[MIDI] No MIDI output ports found. Will write out.mid only.")
        except Exception as e:
            print(f"[MIDI] Could not open MIDI out: {e}. Will write out.mid only.")

    def _record_msg(self, msg) -> None:
        # Convert wall-clock delta to midi ticks
        if self._track is None or self._mid is None:
            return
        now = time.monotonic()
        dt = max(0.0, now - self._last_write_time)
        self._last_write_time = now
        # seconds -> ticks
        tempo = mido.bpm2tempo(self.bpm)  # microsec per beat
        ticks = int(mido.second2tick(dt, self._mid.ticks_per_beat, tempo))
        msg.time = ticks
        self._track.append(msg)

    async def start(self) -> None:
        self._open_midi()
        sub = await self.bus.subscribe(NoteEvent)
        try:
            while True:
                ev = await sub.recv()
                pitch = int(ev.pitch)
                vel = int(max(0, min(127, round(ev.velocity * 127))))
                ch = int(max(0, min(15, ev.channel)))
                dur_s = self._beats_to_seconds(ev.gate_beats)

                if mido is not None:
                    on = mido.Message("note_on", note=pitch, velocity=vel, channel=ch, time=0)
                    off = mido.Message("note_off", note=pitch, velocity=0, channel=ch, time=0)

                    # live send
                    if self._port is not None:
                        self._port.send(on)
                        # schedule note_off without blocking
                        self.scheduler.schedule_in(dur_s, TriggerEvent(name="__midi_note_off__", value=0.0, meta={"msg": off}))
                    # record
                    self._record_msg(on)
                    # record off at correct delay by inserting later (approx via scheduler)
                    self.scheduler.schedule_in(dur_s, TriggerEvent(name="__midi_record_off__", value=0.0, meta={"msg": off}))

        finally:
            await sub.close()

    def close(self) -> None:
        # Write file
        if mido is not None and self._mid is not None:
            try:
                self._mid.save(self.out_path)
                print(f"[MIDI] Wrote {self.out_path}")
            except Exception as e:
                print(f"[MIDI] Failed to save {self.out_path}: {e}")
        # Close port
        try:
            if self._port is not None:
                self._port.close()
        except Exception:
            pass


class MidiAuxRouter(Node):
    """
    Handles scheduled TriggerEvents that carry MIDI messages to send/record,
    so MidiOutSink can schedule without blocking.
    """
    def __init__(self, bus: EventBus, midi_sink: MidiOutSink) -> None:
        self.bus = bus
        self.midi_sink = midi_sink

    async def start(self) -> None:
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name == "__midi_note_off__":
                    msg = ev.meta.get("msg")
                    if msg is not None and self.midi_sink._port is not None:
                        try:
                            self.midi_sink._port.send(msg)
                        except Exception:
                            pass
                elif ev.name == "__midi_record_off__":
                    msg = ev.meta.get("msg")
                    if msg is not None:
                        try:
                            self.midi_sink._record_msg(msg)
                        except Exception:
                            pass
        finally:
            await sub.close()


# =========================
# System
# =========================

class System:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.nodes: List[Node] = []
        self._tasks: List[asyncio.Task] = []

    def add(self, node: Node) -> None:
        self.nodes.append(node)

    async def run_for(self, seconds: float) -> None:
        self._tasks = [asyncio.create_task(n.start()) for n in self.nodes]
        try:
            await asyncio.sleep(max(0.0, seconds))
        finally:
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)


# =========================
# Demo wiring
# =========================

def default_scales() -> ScaleRegistry:
    reg = ScaleRegistry()
    reg.add(Scale("major", (0, 2, 4, 5, 7, 9, 11)))
    reg.add(Scale("minor", (0, 2, 3, 5, 7, 8, 10)))
    reg.add(Scale("dorian", (0, 2, 3, 5, 7, 9, 10)))
    reg.add(Scale("phrygian", (0, 1, 3, 5, 7, 8, 10)))
    reg.add(Scale("lydian", (0, 2, 4, 6, 7, 9, 11)))
    reg.add(Scale("mixolydian", (0, 2, 4, 5, 7, 9, 10)))
    reg.add(Scale("chromatic", tuple(range(12))))
    return reg


async def main() -> None:
    bpm = 124.0
    sys = System()
    bus = sys.bus

    scales = default_scales()

    clock = Clock(bus, bpm=bpm, steps_per_beat=4, loop_steps=16)
    sched = EventScheduler(bus)

    rhythm = EuclideanRhythmNode(
        bus,
        name="euc",
        pattern=EuclideanPattern(steps=16, pulses=5, rotation=0),
        hit_trigger_name="rhythm_hit",
        probability=0.92,
    )

    arp = Arpeggiator(
        bus,
        scales,
        name="arp",
        trigger_name="rhythm_hit",
        scale_name="dorian",
        root_midi=60,
        config=ArpConfig(degrees=(0, 2, 4, 6), pattern="updown", octaves=2, probability=1.0, gate_beats=0.22, velocity=0.85),
    )

    # Continuous modulation -> integrated -> triggers scale changes
    lfo = ContinuousSignal(
        bus,
        name="mod_noise",
        hz=60.0,
        fn=lambda t: 0.50 * math.sin(2.0 * math.pi * 0.41 * t)
                  + 0.35 * math.sin(2.0 * math.pi * 0.73 * t + 1.2)
                  + 0.25 * math.sin(2.0 * math.pi * 1.19 * t + 2.1),
    )
    integ = Integrator(
        bus,
        source_value_name="mod_noise",
        leak=0.06,
        scale=1.0,
        threshold=0.85,
        trigger_name="scale_flip",
        trigger_probability=0.75,
    )

    class ScaleFlipNode(Node):
        def __init__(self, bus: EventBus) -> None:
            self.bus = bus
            self.scales = ["dorian", "phrygian", "lydian", "mixolydian", "minor", "major"]
            self.i = 0
        async def start(self) -> None:
            sub = await self.bus.subscribe(TriggerEvent)
            try:
                while True:
                    ev = await sub.recv()
                    if ev.name != "scale_flip":
                        continue
                    self.i = (self.i + 1) % len(self.scales)
                    await self.bus.publish(ParamChangeEvent(target="arp", param="scale", value=self.scales[self.i]))
            finally:
                await sub.close()

    midi = MidiOutSink(bus, sched, bpm=bpm, out_path="out.mid", preferred_port_contains=None)
    midi_aux = MidiAuxRouter(bus, midi)

    sys.add(sched)
    sys.add(clock)
    sys.add(rhythm)
    sys.add(arp)
    sys.add(lfo)
    sys.add(integ)
    sys.add(ScaleFlipNode(bus))
    sys.add(midi_aux)
    sys.add(midi)

    print("Running for 20 seconds...")
    try:
        await sys.run_for(20.0)
    finally:
        midi.close()


if __name__ == "__main__":
    asyncio.run(main())
