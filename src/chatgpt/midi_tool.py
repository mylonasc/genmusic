from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, List, Tuple, Callable
import threading
import queue
import time
import mido


def _normalize(s: str) -> str:
    return s.strip().lower()


def list_ports() -> dict:
    return {"inputs": mido.get_input_names(), "outputs": mido.get_output_names()}


def _match_score(port_name: str, keywords: Iterable[str]) -> int:
    pn = _normalize(port_name)
    score = 0
    for kw in keywords:
        kw = _normalize(kw)
        if kw and kw in pn:
            score += 1
    return score


def find_best_port(ports: List[str], keywords: Iterable[str]) -> Optional[str]:
    if not ports:
        return None
    scored: List[Tuple[int, str]] = [(_match_score(p, keywords), p) for p in ports]
    scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    best_score, best_port = scored[0]
    return best_port if best_score > 0 else None


@dataclass
class MidiDeviceHandle:
    input_name: Optional[str]
    output_name: Optional[str]
    input: Optional[mido.ports.BaseInput]
    output: Optional[mido.ports.BaseOutput]

    # RX (incoming) thread -> queue
    rx_queue: "queue.Queue[mido.Message]"
    _rx_stop: threading.Event
    _rx_thread: Optional[threading.Thread]

    # TX clock thread (optional)
    _clock_stop: threading.Event
    _clock_thread: Optional[threading.Thread]

    def close(self) -> None:
        # stop clock thread
        self._clock_stop.set()
        if self._clock_thread and self._clock_thread.is_alive():
            self._clock_thread.join(timeout=1.0)

        # stop rx thread
        self._rx_stop.set()
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)

        # close ports
        if self.input is not None:
            self.input.close()
            self.input = None
        if self.output is not None:
            self.output.close()
            self.output = None


class ClockTracker:
    """
    Tracks incoming MIDI clock ticks to estimate BPM.
    MIDI clock sends 24 ticks per quarter note.
    """

    def __init__(self, smoothing: float = 0.2):
        self.smoothing = float(smoothing)
        self._last_tick_t: Optional[float] = None
        self._ema_dt: Optional[float] = None  # EMA of seconds-per-tick

    def update(self, msg: mido.Message, t: Optional[float] = None) -> Optional[float]:
        """
        Call for every incoming msg; only reacts to 'clock'.
        Returns estimated BPM when available, else None.
        """
        if msg.type != "clock":
            return None

        now = time.perf_counter() if t is None else t
        if self._last_tick_t is None:
            self._last_tick_t = now
            return None

        dt = now - self._last_tick_t
        self._last_tick_t = now

        # filter crazy dt (e.g. if clock pauses)
        if dt <= 0 or dt > 1.0:
            self._ema_dt = None
            return None

        if self._ema_dt is None:
            self._ema_dt = dt
        else:
            a = self.smoothing
            self._ema_dt = a * dt + (1 - a) * self._ema_dt

        # BPM = 60 / (seconds per quarter note)
        # seconds per quarter note = seconds_per_tick * 24
        spqn = self._ema_dt * 24.0
        if spqn <= 0:
            return None
        return 60.0 / spqn


class MidiToolNB:
    """
    Non-blocking MIDI tool + MIDI clock send/receive support.
    """

    def __init__(self, default_keywords: Optional[List[str]] = None):
        self.default_keywords = default_keywords or ["digitakt", "elektron"]

    def list(self) -> dict:
        return list_ports()

    def open_device(
        self,
        keywords: Optional[Iterable[str]] = None,
        input_port: Optional[str] = None,
        output_port: Optional[str] = None,
        queue_maxsize: int = 4096,
        poll_sleep: float = 0.001,
    ) -> MidiDeviceHandle:
        ports = list_ports()
        keywords = list(keywords) if keywords is not None else list(self.default_keywords)

        chosen_in = input_port or find_best_port(ports["inputs"], keywords)
        chosen_out = output_port or find_best_port(ports["outputs"], keywords)

        inp = mido.open_input(chosen_in) if chosen_in else None
        outp = mido.open_output(chosen_out) if chosen_out else None

        rx_q: "queue.Queue[mido.Message]" = queue.Queue(maxsize=queue_maxsize)
        rx_stop = threading.Event()
        clock_stop = threading.Event()

        rx_thread = None
        if inp is not None:
            rx_thread = threading.Thread(
                target=self._rx_loop,
                args=(inp, rx_q, rx_stop, poll_sleep),
                daemon=True,
            )
            rx_thread.start()

        return MidiDeviceHandle(
            input_name=chosen_in,
            output_name=chosen_out,
            input=inp,
            output=outp,
            rx_queue=rx_q,
            _rx_stop=rx_stop,
            _rx_thread=rx_thread,
            _clock_stop=clock_stop,
            _clock_thread=None,
        )

    def open_digitakt(self) -> MidiDeviceHandle:
        return self.open_device(keywords=["digitakt", "elektron"])

    @staticmethod
    def _rx_loop(
        inp: mido.ports.BaseInput,
        rx_q: "queue.Queue[mido.Message]",
        stop_event: threading.Event,
        poll_sleep: float,
    ) -> None:
        while not stop_event.is_set():
            msg = inp.poll()
            if msg is None:
                time.sleep(poll_sleep)
                continue
            try:
                rx_q.put_nowait(msg)
            except queue.Full:
                # drop newest if consumer is slow
                pass

    # ---------- Sending general MIDI ----------

    @staticmethod
    def send(handle: MidiDeviceHandle, msg: mido.Message) -> None:
        if handle.output is None:
            raise RuntimeError("No MIDI output port is open on this handle.")
        handle.output.send(msg)

    # Transport helpers
    def send_start(self, handle: MidiDeviceHandle) -> None:
        self.send(handle, mido.Message("start"))

    def send_stop(self, handle: MidiDeviceHandle) -> None:
        self.send(handle, mido.Message("stop"))

    def send_continue(self, handle: MidiDeviceHandle) -> None:
        self.send(handle, mido.Message("continue"))

    # Clock tick (usually you won't call this manually)
    def send_clock_tick(self, handle: MidiDeviceHandle) -> None:
        self.send(handle, mido.Message("clock"))

    # ---------- Receiving (non-blocking) ----------

    @staticmethod
    def get_messages(handle: MidiDeviceHandle, max_messages: int = 256) -> List[mido.Message]:
        msgs: List[mido.Message] = []
        for _ in range(max_messages):
            try:
                msgs.append(handle.rx_queue.get_nowait())
            except queue.Empty:
                break
        return msgs

    @staticmethod
    def filter_messages(msgs: List[mido.Message], msg_type: str) -> List[mido.Message]:
        return [m for m in msgs if m.type == msg_type]

    # ---------- MIDI clock master (computer -> Digitakt) ----------

    def start_clock(
        self,
        handle: MidiDeviceHandle,
        bpm: float,
        send_start: bool = True,
        high_precision: bool = True,
    ) -> None:
        """
        Start sending MIDI clock ticks at `bpm` on a background thread.

        - send_start=True sends 'start' before clock begins.
        - high_precision=True uses perf_counter scheduling (better timing).
        """
        if handle.output is None:
            raise RuntimeError("No MIDI output port is open; cannot send clock.")

        # stop any existing clock thread
        self.stop_clock(handle)

        if send_start:
            self.send_start(handle)

        handle._clock_stop.clear()
        th = threading.Thread(
            target=self._clock_loop,
            args=(handle, float(bpm), handle._clock_stop, high_precision),
            daemon=True,
        )
        handle._clock_thread = th
        th.start()

    def stop_clock(self, handle: MidiDeviceHandle, send_stop: bool = True) -> None:
        """
        Stop the clock thread. Optionally send 'stop'.
        """
        handle._clock_stop.set()
        if handle._clock_thread and handle._clock_thread.is_alive():
            handle._clock_thread.join(timeout=1.0)
        handle._clock_thread = None

        if send_stop and handle.output is not None:
            self.send_stop(handle)

    @staticmethod
    def _clock_loop(
        handle: MidiDeviceHandle,
        bpm: float,
        stop_event: threading.Event,
        high_precision: bool,
    ) -> None:
        # MIDI clock: 24 ticks per quarter note.
        # ticks_per_second = (bpm / 60) * 24
        tps = (bpm / 60.0) * 24.0
        if tps <= 0:
            return
        interval = 1.0 / tps

        if not high_precision:
            # simpler timing
            while not stop_event.is_set():
                handle.output.send(mido.Message("clock"))  # type: ignore[union-attr]
                time.sleep(interval)
            return

        # higher precision scheduling
        next_t = time.perf_counter()
        while not stop_event.is_set():
            now = time.perf_counter()
            if now >= next_t:
                handle.output.send(mido.Message("clock"))  # type: ignore[union-attr]
                next_t += interval
            else:
                # sleep a little but not too much
                time.sleep(min(0.001, next_t - now))

    # ---------- Convenience note/CC senders ----------

    def send_note_on(self, handle: MidiDeviceHandle, note: int, velocity: int = 100, channel: int = 0) -> None:
        self.send(handle, mido.Message("note_on", note=note, velocity=velocity, channel=channel))

    def send_note_off(self, handle: MidiDeviceHandle, note: int, velocity: int = 0, channel: int = 0) -> None:
        self.send(handle, mido.Message("note_off", note=note, velocity=velocity, channel=channel))

    def send_cc(self, handle: MidiDeviceHandle, control: int, value: int, channel: int = 0) -> None:
        self.send(handle, mido.Message("control_change", control=control, value=value, channel=channel))
