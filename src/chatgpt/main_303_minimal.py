#!/usr/bin/env python3
"""
Minimal 303-style pattern with 4-on-the-floor drums.

Features:
- 4 on the floor kick (every beat)
- Simple 303 bass pattern with slides
- Minimal randomness

Usage:
    python3 main_303_minimal.py
"""

import asyncio
import os
from typing import Optional

from genmusic import (
    System,
    EventBus,
    Clock,
    EuclideanPattern,
    EuclideanRhythmNode,
    ParamChangeEvent,
    TriggerEvent,
    NoteEvent,
    FluidSynthSink,
    Node,
    DrumNode,
    DrumConfig,
)


class Simple303(Node):
    """Simple 303-style bass with pattern and slides."""

    def __init__(
        self,
        bus: EventBus,
        *,
        root: int = 45,
        pattern: Optional[list] = None,
    ):
        self.bus = bus
        self.root = root
        # Classic 303 pattern - simple repeating
        self.pattern = pattern or [
            0,
            -1,
            0,
            -1,
            3,
            -1,
            0,
            -1,
            0,
            -1,
            3,
            -1,
            2,
            -1,
            3,
            -1,
        ]
        self._step = 0

    async def start(self) -> None:
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != "seq_trigger":
                    continue

                step = ev.meta.get("step", self._step)
                is_slide = ev.meta.get("slide", False)

                note_idx = step % len(self.pattern)
                note_offset = self.pattern[note_idx]

                # -1 means rest
                if note_offset < 0:
                    self._step += 1
                    continue

                pitch = self.root + note_offset
                gate = 0.15 if is_slide else 0.3

                await self.bus.publish(
                    NoteEvent(
                        pitch=pitch,
                        velocity=0.8,
                        gate_beats=gate,
                        channel=0,
                    )
                )

                self._step += 1
        finally:
            await sub.close()


class SlideGenerator(Node):
    """Generates slide triggers for consecutive notes."""

    def __init__(self, bus: EventBus):
        self.bus = bus

    async def start(self) -> None:
        sub = await self.bus.subscribe(TriggerEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != "seq_trigger":
                    continue

                step = ev.meta.get("step", 0)

                # Slide on certain steps (classic 303 feel)
                if step in [2, 6, 10, 14]:
                    await self.bus.publish(
                        TriggerEvent(
                            name="seq_trigger",
                            value=1.0,
                            meta={"step": step + 1, "slide": True},
                        )
                    )
        finally:
            await sub.close()


async def main() -> None:
    bpm = 135.0
    duration = 16.0

    soundfont_path = os.environ.get(
        "FLUIDSYNTH_SOUNDFONT",
        "/usr/share/sounds/sf2/TimGM6mb.sf2",  # Better electronic sounds
    )
    driver = os.environ.get("FLUIDSYNTH_DRIVER", "alsa")

    bus = EventBus()

    clock = Clock(bus, bpm=bpm, steps_per_beat=4, loop_steps=16)

    # 16-step sequencer trigger (every step)
    seq_rhythm = EuclideanRhythmNode(
        bus,
        name="seq",
        pattern=EuclideanPattern(steps=16, pulses=16),
        hit_trigger_name="seq_trigger",
        probability=1.0,
    )

    slide_gen = SlideGenerator(bus)
    bass = Simple303(bus, root=36)

    # 4 on the floor: kick on beats 0, 4, 8, 12 (every beat)
    drums = DrumNode(
        bus,
        name="drums",
        trigger_name="beat",
        config=DrumConfig(
            kick_pattern=(1, 0, 0, 0, 1, 0, 0, 0, 
                           1, 0, 0, 0, 1, 0, 0, 0),
            snare_pattern=(0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0),
            hihat_pattern=(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            kick_pitch=36,
            snare_pitch=38,
            hihat_pitch=42,
            velocity=0.9,
        ),
    )

    # Trigger drums on every beat (every 4 steps)
    beat_rhythm = EuclideanRhythmNode(
        bus,
        name="beat",
        pattern=EuclideanPattern(steps=16, pulses=4),
        hit_trigger_name="beat",
        probability=1.0,
    )

    # Use TimGM6mb for better 303-like sound
    instruments = {
        0: (0, 35),  # Synth Bass (35 is good for 303)
        9: (0, 0),  # Drums
    }

    synth = FluidSynthSink(
        bus,
        bpm=bpm,
        soundfont_path=soundfont_path,
        driver=driver,
        gain=0.7,
        instruments=instruments,
        reverb=0,
        chorus=0,  # 0 = off
        filter_cutoff=100,
    )

    system = System(bus)
    system.add(clock)
    system.add(beat_rhythm)
    system.add(seq_rhythm)
    system.add(slide_gen)
    system.add(bass)
    system.add(drums)
    system.add(synth)

    print(f"Playing 303 for {duration}s...", flush=True)
    print(f"  BPM: {bpm}", flush=True)
    print(f"  Pattern: 4-on-floor + 303 bass", flush=True)

    try:
        await system.run_for(duration)
    finally:
        synth.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
