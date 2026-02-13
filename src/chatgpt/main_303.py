#!/usr/bin/env python3
"""
303-style generative pattern with accents and slides.

Features:
- Classic 303 acid bass line
- Random accent patterns (emphasized notes)
- Slide patterns (consecutive notes for "squelchy" sound)
- Filter envelope modulation

Requirements:
    pip install pyfluidsynth

Usage:
    python main_303.py
"""

import asyncio
import math
import os
import random
from typing import Optional

from genmusic import (
    System,
    EventBus,
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
    FluidSynthSink,
    Node,
    DrumNode,
    DrumConfig,
)


class AccentNode(Node):
    """Generates accent triggers for 303-style emphasis."""

    def __init__(
        self,
        bus: EventBus,
        *,
        source_trigger: str = "seq_trigger",
        accent_probability: float = 0.3,
    ):
        self.bus = bus
        self.source_trigger = source_trigger
        self.accent_probability = accent_probability
        self.rng = random.Random()

    async def start(self) -> None:
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != self.source_trigger:
                    continue

                step = ev.meta.get("step", 0)
                is_accent = self.rng.random() < self.accent_probability

                await self.bus.publish(
                    TriggerEvent(
                        name="note_trigger",
                        value=1.0 if is_accent else 0.5,
                        meta={"step": step, "accent": is_accent},
                    )
                )
        finally:
            await sub.close()


class SlideNode(Node):
    """Generates slide triggers - plays consecutive notes for 303 squelch."""

    def __init__(
        self,
        bus: EventBus,
        *,
        slide_probability: float = 0.2,
        slide_length: int = 2,
    ):
        self.bus = bus
        self.slide_probability = slide_probability
        self.slide_length = slide_length
        self.rng = random.Random(123)
        self._last_was_slide = False

    async def start(self) -> None:
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != "note_trigger":
                    continue

                step = ev.meta.get("step", 0)

                # Slides typically happen on off-beats or certain steps
                is_slide = (
                    not self._last_was_slide
                    and self.rng.random() < self.slide_probability
                    and step % 4 != 0
                )

                if is_slide:
                    # Schedule additional slides
                    for i in range(1, self.slide_length + 1):
                        await self.bus.publish(
                            TriggerEvent(
                                name="slide_note",
                                value=0.7,
                                meta={"step": step + i, "slide": True},
                            )
                        )
                    self._last_was_slide = True
                else:
                    self._last_was_slide = False
        finally:
            await sub.close()


class TB303Node(Node):
    """Generates classic 303-style bass notes."""

    NOTES = [36, 38, 41, 43, 45, 46, 48]  # Minor pentatonic + half-steps

    def __init__(
        self,
        bus: EventBus,
        *,
        root: int = 36,
        pattern: Optional[list] = None,
    ):
        self.bus = bus
        self.root = root
        self.pattern = pattern or [
            0,
            -1,
            2,
            -1,
            3,
            2,
            -1,
            0,
            -1,
            2,
            -1,
            3,
            -1,
            2,
            3,
            -1,
        ]
        self.rng = random.Random(303)
        self._step = 0

    async def start(self) -> None:
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name not in ("note_trigger", "slide_note"):
                    continue

                step = ev.meta.get("step", self._step)
                is_accent = ev.meta.get("accent", False)
                is_slide = ev.meta.get("slide", False)

                note_idx = step % len(self.pattern)
                note_offset = self.pattern[note_idx]

                if note_offset < 0:
                    self._step += 1
                    continue

                pitch = self.root + note_offset
                velocity = 1.0 if is_accent else 0.7
                gate = 0.2 if is_slide else 0.35

                await self.bus.publish(
                    NoteEvent(
                        pitch=pitch,
                        velocity=velocity,
                        gate_beats=gate,
                        channel=0,
                        meta={"step": step, "accent": is_accent, "slide": is_slide},
                    )
                )

                self._step += 1
        finally:
            await sub.close()


class FilterModulator(Node):
    """Modulates filter cutoff for 303 squelch."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._last_mod = 0

    async def start(self) -> None:
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != "note_trigger":
                    continue

                is_accent = ev.meta.get("accent", False)

                # Accents open the filter
                cutoff = 110 if is_accent else 70

                await self.bus.publish(
                    ParamChangeEvent(
                        target="synth",
                        param="filter",
                        value=cutoff,
                    )
                )
        finally:
            await sub.close()


async def main() -> None:
    bpm = 128.0
    duration = 32.0

    soundfont_path = os.environ.get(
        "FLUIDSYNTH_SOUNDFONT", "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    )
    driver = os.environ.get("FLUIDSYNTH_DRIVER", "alsa")

    if not os.path.exists(soundfont_path):
        print(f"[Warning] SoundFont not found: {soundfont_path}")

    bus = EventBus()

    clock = Clock(bus, bpm=bpm, steps_per_beat=4, loop_steps=16)

    # Main 16-step sequencer trigger
    seq_rhythm = EuclideanRhythmNode(
        bus,
        name="seq",
        pattern=EuclideanPattern(steps=16, pulses=16),  # Every step triggers
        hit_trigger_name="seq_trigger",
        probability=1.0,
    )

    # Add variation - some steps get filtered
    var_rhythm = EuclideanRhythmNode(
        bus,
        name="var",
        pattern=EuclideanPattern(steps=16, pulses=6, rotation=2),
        hit_trigger_name="variation_trigger",
        probability=0.5,
    )

    accent = AccentNode(bus, accent_probability=0.25)
    slides = SlideNode(bus, slide_probability=0.15, slide_length=2)
    bass = TB303Node(bus, root=50)
    filter_mod = FilterModulator(bus)

    # Simple drum pattern
    drum_rhythm = EuclideanRhythmNode(
        bus,
        name="drums",
        pattern=EuclideanPattern(steps=16, pulses=9, rotation=0),
        hit_trigger_name="drum_trigger",
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

    synth = FluidSynthSink(
        bus,
        bpm=bpm,
        soundfont_path=soundfont_path,
        driver=driver,
        gain=0.6,
        preset="303",
        reverb=10,
        chorus=0,
        filter_cutoff=80,
    )

    system = System(bus)
    system.add(clock)
    system.add(seq_rhythm)
    system.add(var_rhythm)
    system.add(accent)
    system.add(slides)
    system.add(bass)
    system.add(filter_mod)
    system.add(drum_rhythm)
    system.add(drums)
    system.add(synth)

    print(f"Playing 303 pattern for {duration}s...", flush=True)
    print(f"  BPM: {bpm}", flush=True)
    print(f"  Pattern: 16-step with accents & slides", flush=True)
    print(f"  Preset: 303", flush=True)

    try:
        await system.run_for(duration)
    finally:
        synth.close()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
