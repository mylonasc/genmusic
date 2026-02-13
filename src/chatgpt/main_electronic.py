#!/usr/bin/env python3
"""
Electronic/303-style generative music with FluidSynth.

Features:
- 303-style lead (square wave)
- Synth bass
- Analog drums
- Real-time filter sweeps
- Modulated reverb/chorus
- Probability-based note generation

Requirements:
    pip install pyfluidsynth

Usage:
    python main_electronic.py
"""

import asyncio
import math
import os
import random

from genmusic import (
    System,
    EventScheduler,
    Clock,
    default_scales,
    EuclideanPattern,
    EuclideanRhythmNode,
    Arpeggiator,
    ArpConfig,
    ContinuousSignal,
    Integrator,
    ParamChangeEvent,
    TriggerEvent,
    NoteEvent,
    EventBus,
    FluidSynthSink,
    Node,
    DrumNode,
    DrumConfig,
    BassNode,
)


class ParamModulator(Node):
    """Modulates synth parameters based on continuous signal."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.filter_val = 80
        self.reverb_val = 20
        self.chorus_val = 0

    async def start(self) -> None:
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != "modulate_params":
                    continue

                val = ev.value if hasattr(ev, "value") else 0.5

                self.filter_val = int(40 + (val + 1) * 40)
                self.reverb_val = int(10 + (val + 1) * 50)
                self.chorus_val = int(max(0, (val - 0.5) * 60))

                await self.bus.publish(
                    ParamChangeEvent(
                        target="synth", param="filter", value=self.filter_val
                    )
                )
                await self.bus.publish(
                    ParamChangeEvent(
                        target="synth", param="reverb", value=self.reverb_val
                    )
                )
                await self.bus.publish(
                    ParamChangeEvent(
                        target="synth", param="chorus", value=self.chorus_val
                    )
                )
        finally:
            await sub.close()


class ProbabilityModulator(Node):
    """Modulates note probability for generative variation."""

    def __init__(self, bus: EventBus, targets: list) -> None:
        self.bus = bus
        self.targets = targets
        self.base_probs = {"lead": 0.9, "bass": 1.0, "chords": 0.5}
        self.current = dict(self.base_probs)

    async def start(self) -> None:
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != "prob_flip":
                    continue

                for target in self.targets:
                    delta = random.choice([-0.15, -0.1, 0.0, 0.1, 0.15, 0.2])
                    new_prob = max(0.1, min(1.0, self.current.get(target, 0.7) + delta))
                    self.current[target] = new_prob

                    await self.bus.publish(
                        ParamChangeEvent(
                            target=target, param="probability", value=new_prob
                        )
                    )
        finally:
            await sub.close()


async def main() -> None:
    bpm = 128.0
    duration = 60.0

    soundfont_path = os.environ.get(
        "FLUIDSYNTH_SOUNDFONT", "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    )
    driver = os.environ.get("FLUIDSYNTH_DRIVER", "alsa")
    preset = os.environ.get("FLUIDSYNTH_PRESET", "303")

    if not os.path.exists(soundfont_path):
        print(f"[Warning] SoundFont not found: {soundfont_path}")

    sys = System()
    bus = sys.bus
    scales = default_scales()

    sched = EventScheduler(bus)
    clock = Clock(bus, bpm=bpm, steps_per_beat=4, loop_steps=16)

    lead_rhythm = EuclideanRhythmNode(
        bus,
        name="lead_rhythm",
        pattern=EuclideanPattern(steps=16, pulses=5, rotation=1),
        hit_trigger_name="lead_trigger",
        probability=0.85,
    )

    bass_rhythm = EuclideanRhythmNode(
        bus,
        name="bass_rhythm",
        pattern=EuclideanPattern(steps=8, pulses=3, rotation=0),
        hit_trigger_name="bass_trigger",
        probability=1.0,
    )

    drums = DrumNode(
        bus,
        name="drums",
        trigger_name="drum_trigger",
        config=DrumConfig(
            kick_pattern=(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
            snare_pattern=(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
            hihat_pattern=(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1),
            kick_pitch=36,
            snare_pitch=38,
            hihat_pitch=42,
            velocity=0.9,
        ),
    )

    bass = BassNode(
        bus,
        scales,
        name="bass",
        trigger_name="bass_trigger",
        scale_name="dorian",
        root_midi=36,
        pattern="walk",
        probability=1.0,
        gate_beats=0.35,
        velocity=0.8,
        channel=1,
    )

    lead_arp = Arpeggiator(
        bus,
        scales,
        name="lead",
        trigger_name="lead_trigger",
        scale_name="dorian",
        root_midi=60,
        config=ArpConfig(
            degrees=(0, 3, 5, 7, 10),
            pattern="random",
            octaves=1,
            probability=0.9,
            gate_beats=0.15,
            velocity=0.75,
            channel=0,
        ),
    )

    chord_arp = Arpeggiator(
        bus,
        scales,
        name="chords",
        trigger_name="lead_trigger",
        scale_name="dorian",
        root_midi=48,
        config=ArpConfig(
            degrees=(0, 3, 7),
            pattern="up",
            octaves=1,
            probability=0.5,
            gate_beats=0.4,
            velocity=0.5,
            channel=2,
        ),
    )

    rng = random.Random(42)
    lfo_state = {"x": 0.5}

    def chaotic_lfo(t: float) -> float:
        lfo_state["x"] += rng.uniform(-0.12, 0.12)
        lfo_state["x"] = max(-1.0, min(1.0, lfo_state["x"]))
        base = 0.5 * math.sin(2.0 * math.pi * 0.2 * t)
        mod = 0.3 * math.sin(2.0 * math.pi * 0.7 * t + 2.0)
        noise = 0.2 * lfo_state["x"]
        return base + mod + noise

    lfo = ContinuousSignal(
        bus,
        name="mod",
        hz=20.0,
        fn=chaotic_lfo,
    )

    integ_filter = Integrator(
        bus,
        source_value_name="mod",
        leak=0.05,
        scale=1.0,
        threshold=0.7,
        trigger_name="modulate_params",
        trigger_probability=0.8,
    )

    integ_prob = Integrator(
        bus,
        source_value_name="mod",
        leak=0.08,
        scale=0.6,
        threshold=0.4,
        trigger_name="prob_flip",
        trigger_probability=0.5,
    )

    param_mod = ParamModulator(bus)
    prob_mod = ProbabilityModulator(bus, targets=["lead", "bass", "chords"])

    synth = FluidSynthSink(
        bus,
        bpm=bpm,
        soundfont_path=soundfont_path,
        driver=driver,
        gain=0.5,
        preset=preset,
        reverb=20,
        chorus=0,
        filter_cutoff=80,
    )

    sys.add(clock)
    sys.add(drums)
    sys.add(bass_rhythm)
    sys.add(lead_rhythm)
    sys.add(bass)
    sys.add(lead_arp)
    sys.add(chord_arp)
    sys.add(lfo)
    sys.add(integ_filter)
    sys.add(integ_prob)
    sys.add(param_mod)
    sys.add(prob_mod)
    sys.add(synth)

    print(f"Playing for {duration}s via FluidSynth ({preset} preset)...")
    print(f"  SoundFont: {soundfont_path}")
    print(f"  Driver: {driver}")
    print(f"  BPM: {bpm}")
    print("  Layers: Drums (ch9) | Bass (ch1) | Lead (ch0) | Chords (ch2)")
    print("  Real-time: Filter sweeps, reverb, probability modulation")

    try:
        await sys.run_for(duration)
    finally:
        synth.close()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
