#!/usr/bin/env python3
"""
FluidSynth example - elaborate generative music with drums, bass, and arps.

Requirements:
    pip install pyfluidsynth

Usage:
    python main_fluidsynth.py

The soundfont path defaults to /usr/share/sounds/sf2/FluidR3_GM.sf2
(on Linux). On macOS/Windows, adjust soundfont_path below.

What it does:
- Drum pattern on channel 9 (GM drums)
- Bass line with pattern variation
- Two arpeggiators (melody + chords)
- Random modulation of patterns and scales
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


class ScaleFlipNode(Node):
    def __init__(self, bus: EventBus, targets: list) -> None:
        self.bus = bus
        self.targets = targets
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
                scale = self.scales[self.i]
                for target in self.targets:
                    await self.bus.publish(
                        ParamChangeEvent(target=target, param="scale", value=scale)
                    )
        finally:
            await sub.close()


class PatternModulator(Node):
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.bass_patterns = ["root", "octave", "fifth", "walk"]
        self.pattern_idx = 0

    async def start(self) -> None:
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != "pattern_flip":
                    continue
                self.pattern_idx = (self.pattern_idx + 1) % len(self.bass_patterns)
                await self.bus.publish(
                    ParamChangeEvent(
                        target="bass",
                        param="pattern",
                        value=self.bass_patterns[self.pattern_idx],
                    )
                )
        finally:
            await sub.close()


async def main() -> None:
    bpm = 124.0
    duration = 60.0

    soundfont_path = os.environ.get(
        "FLUIDSYNTH_SOUNDFONT", "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    )
    driver = os.environ.get("FLUIDSYNTH_DRIVER", "alsa")

    if not os.path.exists(soundfont_path):
        print(f"[Warning] SoundFont not found: {soundfont_path}")
        print("Set FLUIDSYNTH_SOUNDFONT environment variable to your .sf2 file")

    sys = System()
    bus = sys.bus
    scales = default_scales()

    sched = EventScheduler(bus)
    clock = Clock(bus, bpm=bpm, steps_per_beat=4, loop_steps=16)

    mel_rhythm = EuclideanRhythmNode(
        bus,
        name="mel_rhythm",
        pattern=EuclideanPattern(steps=16, pulses=7, rotation=3),
        hit_trigger_name="mel_trigger",
        probability=0.9,
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
            kick_pattern=(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0),
            snare_pattern=(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
            hihat_pattern=(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1),
            kick_pitch=36,
            snare_pitch=38,
            hihat_pitch=42,
            velocity=0.85,
        ),
    )

    bass = BassNode(
        bus,
        scales,
        name="bass",
        trigger_name="bass_trigger",
        scale_name="dorian",
        root_midi=36,
        pattern="octave",
        probability=1.0,
        gate_beats=0.4,
        velocity=0.75,
        channel=1,
    )

    mel_arp = Arpeggiator(
        bus,
        scales,
        name="mel_arp",
        trigger_name="mel_trigger",
        scale_name="dorian",
        root_midi=60,
        config=ArpConfig(
            degrees=(0, 2, 4, 7, 9),
            pattern="updown",
            octaves=2,
            probability=0.85,
            gate_beats=0.18,
            velocity=0.7,
            channel=0,
        ),
    )

    chord_arp = Arpeggiator(
        bus,
        scales,
        name="chord_arp",
        trigger_name="mel_trigger",
        scale_name="dorian",
        root_midi=48,
        config=ArpConfig(
            degrees=(0, 4, 7),
            pattern="up",
            octaves=1,
            probability=0.6,
            gate_beats=0.35,
            velocity=0.5,
            channel=2,
        ),
    )

    rng = random.Random(42)
    lfo_state = {"x": 0.0}

    def modulated_lfo(t: float) -> float:
        lfo_state["x"] += rng.uniform(-0.8, 0.8)
        lfo_state["x"] = max(-1.0, min(1.0, lfo_state["x"]))
        return lfo_state["x"]

    lfo = ContinuousSignal(
        bus,
        name="mod",
        hz=400.0,
        fn=lambda t: 0.6 * math.sin(2.0 * math.pi * 0.3 * t)
        + 0.3 * math.sin(2.0 * math.pi * 0.7 * t + 1.0)
        + 0.1 * modulated_lfo(t),
    )

    integ = Integrator(
        bus,
        source_value_name="mod",
        leak=0.04,
        scale=1.2,
        threshold=0.9,
        trigger_name="scale_flip",
        trigger_probability=0.6,
    )

    integ2 = Integrator(
        bus,
        source_value_name="mod",
        leak=0.04,
        scale=0.9,
        threshold=0.3,
        trigger_name="pattern_flip",
        trigger_probability=0.6,
    )

    scale_flip = ScaleFlipNode(bus, targets=["mel_arp", "chord_arp", "bass"])
    pattern_mod = PatternModulator(bus)

    # GM instrument programs: (bank, program)
    # See https://www.midi.org/specifications/midi-reference-tables/gm-level-1-program-chart
    instruments = {
        0: (0, 0),  # ch0: Acoustic Grand Piano
        1: (0, 33),  # ch1: Electric Bass (finger)
        2: (0, 48),  # ch2: Strings
        9: (0, 0),  # ch9: Standard Drum Kit
    }

    synth = FluidSynthSink(
        bus,
        bpm=bpm,
        soundfont_path=soundfont_path,
        driver=driver,
        gain=0.5,
        instruments=instruments,
    )

    sys.add(clock)
    sys.add(drums)
    sys.add(bass_rhythm)
    sys.add(mel_rhythm)
    sys.add(bass)
    sys.add(mel_arp)
    sys.add(chord_arp)
    sys.add(lfo)
    sys.add(integ)
    sys.add(integ2)
    sys.add(scale_flip)
    sys.add(pattern_mod)
    sys.add(synth)

    print(f"Playing for {duration} seconds via FluidSynth...")
    print(f"  SoundFont: {soundfont_path}")
    print(f"  Driver: {driver}")
    print(f"  BPM: {bpm}")
    print("  Layers: Drums (ch9) | Bass (ch1) | Melody (ch0) | Chords (ch2)")

    try:
        await sys.run_for(duration)
    finally:
        synth.close()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
