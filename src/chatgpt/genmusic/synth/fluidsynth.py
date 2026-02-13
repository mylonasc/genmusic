"""
FluidSynth audio output.
"""

import asyncio
from typing import Dict, Optional, Tuple

from ..events import EventBus, NoteEvent, ParamChangeEvent
from ..nodes.base import Node
from .presets import INSTRUMENT_PRESETS, get_preset


class FluidSynthSink(Node):
    """
    Plays NoteEvent via FluidSynth software synthesizer.
    Supports multiple channels with effects, filters, and instrument presets.
    Requires: pip install pyfluidsynth

    Real-time controls via ParamChangeEvent:
    - target="synth", param="reverb", value=0-127
    - target="synth", param="chorus", value=0-127
    - target="synth", param="filter", value=0-127 (cutoff)
    - target="synth", param="volume", value=0-127
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        bpm: float,
        soundfont_path: str = "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        driver: str = "alsa",
        gain: float = 0.5,
        preset: str = "acoustic",
        instruments: Optional[Dict[int, Tuple[int, int]]] = None,
        reverb: int = 30,
        chorus: int = 0,
        filter_cutoff: int = 127,
    ):
        self.bus = bus
        self.bpm = bpm
        self._fs = None
        self._sfid = None

        self.instruments = instruments or get_preset(preset)
        self.reverb = reverb
        self.chorus = chorus
        self.filter_cutoff = filter_cutoff

        try:
            import fluidsynth

            self._fs = fluidsynth.Synth()
            self._fs.start(driver=driver)
            self._fs.setting("synth.gain", gain)
            self._sfid = self._fs.sfload(soundfont_path)

            for ch, (bank, prog) in self.instruments.items():
                if ch == 9:
                    # Drums use GM drum bank (128)
                    self._fs.program_select(ch, self._sfid, 128, 0)
                else:
                    self._fs.program_select(ch, self._sfid, bank, prog)

            self._apply_reverb()
            self._apply_chorus()
            self._apply_filter_all()

            print(
                f"[FluidSynthSink] Initialized with preset '{preset}': {self.instruments}"
            )
        except Exception as e:
            print(f"[FluidSynthSink] Failed to initialize: {e}")
            self._fs = None

    def _apply_reverb(self) -> None:
        if self._fs:
            self._fs.set_reverb(self.reverb / 127.0, 0.2, 0.5, 1.0)

    def _apply_chorus(self) -> None:
        if self._fs:
            if self.chorus > 0:
                self._fs.set_chorus(3, self.chorus / 127.0 * 3.0, 0.3, 1)
            else:
                self._fs.set_chorus(0, 0, 0, 0)

    def _apply_filter_all(self) -> None:
        if self._fs:
            for ch in range(16):
                self._fs.cc(ch, 74, self.filter_cutoff)

    def _beats_to_seconds(self, beats: float) -> float:
        return (60.0 / max(1e-9, self.bpm)) * float(beats)

    async def start(self) -> None:
        if self._fs is None:
            return

        note_sub = await self.bus.subscribe(NoteEvent)
        param_sub = await self.bus.subscribe(ParamChangeEvent)

        try:
            while True:
                done, _ = await asyncio.wait(
                    [
                        asyncio.create_task(note_sub.recv()),
                        asyncio.create_task(param_sub.recv()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    ev = task.result()

                    if isinstance(ev, NoteEvent):
                        pitch = int(ev.pitch)
                        vel = int(max(0, min(127, round(ev.velocity * 127))))
                        ch = int(ev.channel) if hasattr(ev, "channel") else 0
                        dur_s = self._beats_to_seconds(ev.gate_beats)

                        self._fs.noteon(ch, pitch, vel)
                        asyncio.create_task(self._noteoff(ch, pitch, dur_s))

                    elif isinstance(ev, ParamChangeEvent):
                        if ev.target != "synth":
                            continue
                        if ev.param == "reverb":
                            self.reverb = int(max(0, min(127, ev.value)))
                            self._apply_reverb()
                        elif ev.param == "chorus":
                            self.chorus = int(max(0, min(127, ev.value)))
                            self._apply_chorus()
                        elif ev.param == "filter":
                            self.filter_cutoff = int(max(0, min(127, ev.value)))
                            self._apply_filter_all()
                        elif ev.param == "volume":
                            vol = int(max(0, min(127, ev.value)))
                            for ch in range(16):
                                self._fs.cc(ch, 7, vol)
        finally:
            await note_sub.close()
            await param_sub.close()

    async def _noteoff(self, ch: int, pitch: int, dur_s: float) -> None:
        await asyncio.sleep(max(0.01, dur_s))
        if self._fs:
            self._fs.noteoff(ch, pitch)

    def close(self) -> None:
        if self._fs:
            try:
                self._fs.delete()
            except Exception:
                pass
            self._fs = None
