# Generative Algorithmic Music Control Framework (Python / asyncio)

This is a small event-driven framework for **generative / algorithmic music control** in Python.

It focuses on:
- **Discrete events** (triggers, ticks, parameter changes)
- **Continuous control signals** (ValueEvents)
- **Bridging continuous → discrete** via integration (Integrator)
- Musical building blocks:
  - **Scale registry** (arbitrary many scales)
  - **Arpeggiator** (event-driven changes like scale/pattern)
  - **Euclidean rhythms** (Bjorklund)
  - **Randomness policies** (probabilistic event consumption)
- **Non-blocking** execution using `asyncio`
- Optional **MIDI output** (so you can “hear it” right away)

> This library is intentionally modular: you wire nodes together with an `EventBus` and run them concurrently.

---

## 1) Install & Run (with MIDI output)

### Requirements
- Python **3.10+**
- If you want MIDI output:
  - `mido`
  - `python-rtmidi`

Install:
```bash
pip install mido python-rtmidi
````

Run the demo script (the single Python file you have):

```bash
python magicbus_demo.py
```

What happens:

* If the system finds a MIDI output port, it streams notes live.
* It also writes a file called `out.mid` on exit.
* If no MIDI port is available, it will still write `out.mid` (open it in a DAW/player).

### If you hear nothing

You likely don’t have a synth connected to your MIDI output.

Platform tips:

* **macOS**: enable *Audio MIDI Setup → IAC Driver*, route to a synth/DAW (GarageBand, Ableton, etc.)
* **Windows**: use loopMIDI + a softsynth/DAW
* **Linux**: route to fluidsynth or a DAW via ALSA/JACK

---

## 2) Core Concepts

### Event-driven design

Everything communicates via an `EventBus`.

Common event types:

* `TickEvent`: periodic grid timing (from `Clock`)
* `TriggerEvent`: discrete impulses (rhythm hits, gates, “bangs”)
* `ValueEvent`: continuous (or sampled) control values
* `ParamChangeEvent`: runtime parameter updates (e.g., change scale)
* `NoteEvent`: emitted notes (no audio built-in; you route to MIDI or a synth later)

### Nodes

A **Node** is an async task that subscribes to certain events and publishes others.

Examples:

* `Clock` publishes `TickEvent`
* `EuclideanRhythmNode` consumes `TickEvent`, publishes `TriggerEvent`
* `Arpeggiator` consumes `TriggerEvent`, publishes `NoteEvent`
* `Integrator` consumes `ValueEvent`, publishes `TriggerEvent` (when thresholds hit)
* `MidiOutSink` consumes `NoteEvent`, sends MIDI messages and writes a MIDI file

---

## 3) Minimal Example: Euclidean Rhythm → Arpeggiator → MIDI

This is the simplest “I want to hear notes now” wiring:

```python
import asyncio
from magicbus import (
    System, EventScheduler, Clock,
    default_scales, EuclideanPattern, EuclideanRhythmNode,
    Arpeggiator, ArpConfig,
    MidiOutSink, MidiAuxRouter
)

async def main():
    bpm = 124.0
    sys = System()
    bus = sys.bus

    scales = default_scales()

    sched = EventScheduler(bus)
    clock = Clock(bus, bpm=bpm, steps_per_beat=4, loop_steps=16)

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
        config=ArpConfig(
            degrees=(0, 2, 4, 6),
            pattern="updown",
            octaves=2,
            gate_beats=0.22,
            velocity=0.85,
        ),
    )

    midi = MidiOutSink(bus, sched, bpm=bpm, out_path="out.mid")
    midi_aux = MidiAuxRouter(bus, midi)

    sys.add(sched)
    sys.add(clock)
    sys.add(rhythm)
    sys.add(arp)
    sys.add(midi_aux)
    sys.add(midi)

    try:
        await sys.run_for(15.0)
    finally:
        midi.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4) Continuous → Discrete: Integrator triggers scale changes

This demonstrates your “integrate continuous signals into events” requirement.

Idea:

1. A `ContinuousSignal` emits a modulation value (like a slow LFO or noise).
2. `Integrator` accumulates it and emits a trigger when it crosses a threshold.
3. A small node listens for that trigger and sends a `ParamChangeEvent` to the arpeggiator.

```python
import asyncio
import math
from magicbus import (
    System, EventScheduler, Clock,
    default_scales,
    EuclideanPattern, EuclideanRhythmNode,
    Arpeggiator, ArpConfig,
    ContinuousSignal, Integrator,
    ParamChangeEvent, TriggerEvent,
    MidiOutSink, MidiAuxRouter, EventBus
)

class ScaleFlipNode:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.scales = ["dorian", "phrygian", "lydian", "mixolydian", "minor", "major"]
        self.i = 0

    async def start(self):
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != "scale_flip":
                    continue
                self.i = (self.i + 1) % len(self.scales)
                await self.bus.publish(
                    ParamChangeEvent(target="arp", param="scale", value=self.scales[self.i])
                )
        finally:
            await sub.close()

async def main():
    bpm = 120
    sys = System()
    bus = sys.bus
    scales = default_scales()

    sched = EventScheduler(bus)
    clock = Clock(bus, bpm=bpm, steps_per_beat=4, loop_steps=16)

    rhythm = EuclideanRhythmNode(
        bus,
        pattern=EuclideanPattern(steps=16, pulses=5, rotation=0),
        hit_trigger_name="rhythm_hit",
        probability=0.9,
    )

    arp = Arpeggiator(
        bus, scales,
        name="arp",
        trigger_name="rhythm_hit",
        scale_name="dorian",
        root_midi=60,
        config=ArpConfig(degrees=(0,2,4,6), pattern="up", octaves=2),
    )

    # continuous control signal
    lfo = ContinuousSignal(
        bus,
        name="mod_noise",
        hz=60,
        fn=lambda t: 0.6*math.sin(2*math.pi*0.43*t) + 0.4*math.sin(2*math.pi*0.78*t + 1.1),
    )

    # integrate; trigger when integrated state crosses threshold
    integ = Integrator(
        bus,
        source_value_name="mod_noise",
        leak=0.06,
        scale=1.0,
        threshold=0.85,
        trigger_name="scale_flip",
        trigger_probability=0.75,   # randomness: not every threshold crossing triggers
    )

    midi = MidiOutSink(bus, sched, bpm=bpm, out_path="out.mid")
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

    try:
        await sys.run_for(20.0)
    finally:
        midi.close()

asyncio.run(main())
```

---

## 5) Change parameters live (event-driven)

Use `ParamChangeEvent(target=..., param=..., value=...)`.

### Change the Euclidean rhythm

```python
await bus.publish(ParamChangeEvent(target="euc", param="pulses", value=7))
await bus.publish(ParamChangeEvent(target="euc", param="rotation", value=3))
await bus.publish(ParamChangeEvent(target="euc", param="probability", value=0.7))
```

### Change the arpeggiator behavior

```python
await bus.publish(ParamChangeEvent(target="arp", param="scale", value="minor"))
await bus.publish(ParamChangeEvent(target="arp", param="pattern", value="random"))
await bus.publish(ParamChangeEvent(target="arp", param="degrees", value=[0, 2, 4, 7]))
await bus.publish(ParamChangeEvent(target="arp", param="octaves", value=3))
await bus.publish(ParamChangeEvent(target="arp", param="gate_beats", value=0.15))
```

> Parameters are applied immediately the next time the node handles events.

---

## 6) Customize scales (arbitrarily many)

A scale is defined by semitone offsets within an octave.

Example: add harmonic minor:

```python
from magicbus import Scale

scales.add(Scale("harmonic_minor", (0, 2, 3, 5, 7, 8, 11)))
```

Then switch to it at runtime:

```python
await bus.publish(ParamChangeEvent(target="arp", param="scale", value="harmonic_minor"))
```

---

## 7) Randomness controls

### Probabilistic rhythm hits

`EuclideanRhythmNode(probability=0.8)` means only 80% of hits become triggers.

### Probabilistic note generation

`ArpConfig(probability=0.7)` means only 70% of incoming triggers produce notes.

### Probabilistic integrated triggers

`Integrator(trigger_probability=0.6)` means only 60% of threshold crossings become triggers.

### Swap RandomPolicy

If you want custom randomness (e.g., weighted choices or stateful probability),
implement `RandomPolicy` and pass it into nodes.

---

## 8) Design Notes / Extending

### Add a new node

Create a class with:

* `async def start(self) -> None`
* subscribe to the event types you care about
* publish new events as needed

Example: “transpose all notes by +12”

```python
from magicbus import NoteEvent

class TransposeNode:
    def __init__(self, bus, semis=12):
        self.bus = bus
        self.semis = semis

    async def start(self):
        sub = await self.bus.subscribe(NoteEvent)
        try:
            while True:
                ev = await sub.recv()
                await self.bus.publish(
                    NoteEvent(
                        pitch=ev.pitch + self.semis,
                        velocity=ev.velocity,
                        gate_beats=ev.gate_beats,
                        channel=ev.channel,
                        meta={**ev.meta, "transposed": self.semis},
                    )
                )
        finally:
            await sub.close()
```

Wire it between arp and MIDI sink by just adding it as another subscriber/publisher.

### Add new musical structures

Common next steps:

* chord generators (emit sets of degrees)
* polyrhythm/multi-clock support
* swing/humanization node (perturb trigger timing using `EventScheduler`)
* probability “consumption” semantics (e.g., a trigger can be consumed by one of many nodes)

---

## 9) Troubleshooting

### “No MIDI output ports found”

That’s OK: you’ll still get `out.mid`. Open it in your DAW.

### “Still no sound”

You need a synth attached to a MIDI port. Common solutions:

* DAW with a virtual instrument
* a lightweight softsynth
* OS-native routing (IAC on macOS, loopMIDI on Windows)

### Timing is jittery

This uses `asyncio.sleep()` and Python timing, which is fine for *control-rate prototyping*.
For tighter musical timing, keep the same architecture but:

* use a more accurate scheduler
* or drive the control system from an audio callback / dedicated timing thread

---

## 10) Quick reference

### Events

* `TickEvent(step, beat, steps_per_beat)`
* `TriggerEvent(name, value)`
* `ValueEvent(name, value)`
* `ParamChangeEvent(target, param, value)`
* `NoteEvent(pitch, velocity, gate_beats, channel)`

### Nodes included

* Timing: `Clock`, `EventScheduler`
* Control: `ContinuousSignal`, `Integrator`
* Musical: `EuclideanRhythmNode`, `Arpeggiator`
* Output: `MidiOutSink`, `MidiAuxRouter`

---

If you want, describe your intended “patch style” (e.g., modular synth patching, Live coding, DAW integration),
and I can expand the docs with a recommended folder layout, testing strategy, and a few reusable nodes
(quantizer, chord engine, swing/humanizer, probability router, event recorder).

```
```
