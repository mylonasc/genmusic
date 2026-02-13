"""
Drop-in replacement example that should PLAY continuously via FluidSynth
(using your known-good settings: fs.start(driver='pulseaudio')).

What it does:
- Uses Euclidean rhythms to drive TWO arpeggiators (melody + bass) for richer output.
- A continuous random-walk signal modulates:
  - melody arp pattern (up/down/updown/random)
  - melody gate length (phrasing)
  - occasional scale changes (harmonic motion)
- Uses a corrected FluidSynth sink that avoids NoteEvent class-mismatch by injecting NoteEventType.

Prereqs:
    pip install pyfluidsynth
    (and your SoundFont exists at /usr/share/sounds/sf2/FluidR3_GM.sf2)

Assumes your framework file is importable as: genmusic
"""

import asyncio
import random
import time
from typing import Optional

import fluidsynth  # pyfluidsynth

from genmusic import (
    System,
    Clock,
    default_scales,
    EuclideanRhythmNode,
    EuclideanPattern,
    Arpeggiator,
    ArpConfig,
    ValueEvent,
    ParamChangeEvent,
    NoteEvent,
)


class FluidSynthSink:
    """
    Corrected FluidSynth sink:
    - Uses your known-good init pattern:
        fs = fluidsynth.Synth()
        fs.start(driver='pulseaudio')
        sfid = fs.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")
        fs.program_select(0, sfid, 0, 0)
    - Subscribes to the EXACT NoteEvent class instance used by the publishers by injecting NoteEventType.
    - Schedules note-offs non-blocking via asyncio.
    """

    def __init__(
        self,
        bus,
        *,
        NoteEventType,
        bpm: float,
        soundfont_path: str = "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        driver: str = "pulseaudio",
        channel: int = 0,
        bank: int = 0,
        program: int = 0,
        debug: bool = False,
    ):
        self.bus = bus
        self.NoteEventType = NoteEventType
        self.bpm = float(bpm)
        self.debug = debug

        self.channel = int(channel)
        self.bank = int(bank)
        self.program = int(program)

        # 1) Initialize
        self.fs = fluidsynth.Synth()

        # 2) Start driver (your known-good setting)
        self.fs.start(driver=driver)

        # 3) Load SoundFont + select program
        self.sfid = self.fs.sfload(soundfont_path)
        self.fs.program_select(self.channel, self.sfid, self.bank, self.program)

        if self.debug:
            # quick audible sanity check (C4 blip)
            self.fs.noteon(self.channel, 60, 100)
            # schedule note off without blocking asyncio startup
            import threading

            def _off():
                time.sleep(0.25)
                self.fs.noteoff(self.channel, 60)

            threading.Thread(target=_off, daemon=True).start()
            print("[FluidSynthSink] Initialized OK (played a short C4 blip).")

    def _beats_to_seconds(self, beats: float) -> float:
        return (60.0 / max(1e-9, self.bpm)) * float(beats)

    async def start(self) -> None:
        sub = await self.bus.subscribe(self.NoteEventType)
        try:
            while True:
                ev = await sub.recv()

                pitch = int(ev.pitch)
                vel = int(max(0, min(127, round(float(ev.velocity) * 127))))
                ch = int(getattr(ev, "channel", self.channel))

                gate_beats = float(getattr(ev, "gate_beats", getattr(ev, "gate", 0.25)))
                dur_s = self._beats_to_seconds(gate_beats)

                if self.debug:
                    print(f"[FluidSynthSink] noteon ch={ch} pitch={pitch} vel={vel} dur_s={dur_s:.3f}")

                self.fs.noteon(ch, pitch, vel)
                asyncio.create_task(self._noteoff_later(ch, pitch, dur_s))
        finally:
            await sub.close()

    async def _noteoff_later(self, ch: int, pitch: int, dur_s: float) -> None:
        await asyncio.sleep(max(0.01, dur_s))
        self.fs.noteoff(ch, pitch)

    def close(self) -> None:
        try:
            self.fs.delete()
        except Exception:
            pass


class MusicalModulator:
    """
    Listens to ValueEvent(name='rand') and modulates:
    - mel_arp pattern
    - mel_arp gate length
    - occasional scale change for BOTH melody and bass
    """

    def __init__(self, bus):
        self.bus = bus
        self._last_fast_change = 0.0
        self._last_scale_change = 0.0
        self._scale_cycle = ["dorian", "mixolydian", "lydian", "minor"]

    async def start(self) -> None:
        sub = await self.bus.subscribe(ValueEvent)
        try:
            while True:
                ev = await sub.recv()
                if ev.name != "rand":
                    continue

                now = time.monotonic()
                v = float(ev.value)

                # fast-ish changes (pattern + phrasing), rate limited
                if now - self._last_fast_change > 0.30:
                    if v > 0.65:
                        pat = "up"
                    elif v < -0.65:
                        pat = "down"
                    else:
                        pat = "random" if (int(now * 2) % 4 == 0) else "updown"

                    # phrasing: shorter vs longer notes
                    gate = 0.12 if v > 0.3 else (0.22 if v < -0.3 else 0.16)

                    await self.bus.publish(ParamChangeEvent(target="mel_arp", param="pattern", value=pat))
                    await self.bus.publish(ParamChangeEvent(target="mel_arp", param="gate_beats", value=gate))

                    self._last_fast_change = now

                # slower harmonic motion (scale change), only when v is "extreme"
                if now - self._last_scale_change > 3.0 and abs(v) > 0.75:
                    idx = int(((v + 1.0) / 2.0) * (len(self._scale_cycle) - 1))
                    scale = self._scale_cycle[idx]

                    await self.bus.publish(ParamChangeEvent(target="mel_arp", param="scale", value=scale))
                    await self.bus.publish(ParamChangeEvent(target="bass_arp", param="scale", value=scale))

                    self._last_scale_change = now
        finally:
            await sub.close()


async def main() -> None:
    bpm = 120.0

    sys = System()
    bus = sys.bus

    # Clock drives TickEvent; Euclidean nodes subscribe internally
    clock = Clock(bus, bpm=bpm, steps_per_beat=4, loop_steps=16)

    scales = default_scales()

    # Two Euclidean rhythms = groove + variation
    mel_rhythm = EuclideanRhythmNode(
        bus,
        name="mel_rhy",
        pattern=EuclideanPattern(steps=16, pulses=11, rotation=2),
        hit_trigger_name="mel_hit",
        probability=0.95,
    )

    bass_rhythm = EuclideanRhythmNode(
        bus,
        name="bass_rhy",
        pattern=EuclideanPattern(steps=16, pulses=5, rotation=0),
        hit_trigger_name="bass_hit",
        probability=0.98,
    )

    # Melody arp: wider pool + more octaves
    mel_arp = Arpeggiator(
        bus,
        scales,
        name="mel_arp",
        trigger_name="mel_hit",
        scale_name="dorian",
        root_midi=60,  # C4
        config=ArpConfig(
            degrees=(0, 2, 4, 6, 7, 9),
            pattern="updown",
            octaves=3,
            probability=1.0,
            gate_beats=0.16,
            velocity=0.88,
            channel=0,
        ),
    )

    # Bass arp: slower, lower register
    bass_arp = Arpeggiator(
        bus,
        scales,
        name="bass_arp",
        trigger_name="bass_hit",
        scale_name="dorian",
        root_midi=36,  # C2
        config=ArpConfig(
            degrees=(0, 4, 7, 9),
            pattern="up",
            octaves=1,
            probability=1.0,
            gate_beats=0.30,
            velocity=0.75,
            channel=0,
        ),
    )

    # Random continuous control: smooth random walk in [-1, 1]
    rng = random.Random(7)
    state = {"x": 0.0}

    def randwalk(_t: float) -> float:
        x = state["x"] + rng.uniform(-0.10, 0.10)
        x = max(-1.0, min(1.0, x))
        state["x"] = x
        return x

    # NOTE: uses your framework ContinuousSignal
    from genmusic import ContinuousSignal

    rand_signal = ContinuousSignal(bus, name="rand", fn=randwalk, hz=15.0)
    modulator = MusicalModulator(bus)

    # FluidSynth sink (your known-good driver setting)
    sink = FluidSynthSink(
        bus,
        NoteEventType=NoteEvent,  # IMPORTANT: avoids class mismatch
        bpm=bpm,
        driver="pulseaudio",
        soundfont_path="/usr/share/sounds/sf2/FluidR3_GM.sf2",
        program=0,    # acoustic grand
        channel=0,
        debug=False,  # set True to print noteon logs + play init blip
    )

    sys.add(clock)
    sys.add(mel_rhythm)
    sys.add(bass_rhythm)
    sys.add(mel_arp)
    sys.add(bass_arp)
    sys.add(rand_signal)
    sys.add(modulator)
    sys.add(sink)

    print("Playing for 30 seconds (melody + bass, random modulation of pattern/gate/scale)...")
    try:
        await sys.run_for(30.0)
    finally:
        sink.close()


if __name__ == "__main__":
    asyncio.run(main())
