
# MidiToolNB — Non-Blocking MIDI I/O + MIDI Clock Sync (Digitakt + others)

A small Python utility for discovering a MIDI device (e.g., **Elektron Digitakt**) and doing **non-blocking** MIDI I/O, including **MIDI clock + transport** (start/stop/continue) for synchronization.

It uses:
- `mido` for MIDI message objects + port opening
- `python-rtmidi` as the backend
- a **background RX thread** that polls input and pushes messages into a **queue**
- an optional **clock TX thread** that generates MIDI clock ticks at a given BPM

---

## Install

```bash
pip install mido python-rtmidi
````

---

## File Layout

* `midi_tool_nb_clock.py` — contains:

  * `MidiToolNB` (the tool class)
  * `MidiDeviceHandle` (opened ports + threads + queue)
  * `ClockTracker` (optional helper for estimating BPM from incoming clock)

---

## What “Non-Blocking” Means Here

Classic MIDI reads like:

```python
for msg in inport:
    ...
```

**block** the current thread. If that’s your main loop (UI/audio/game loop), it can freeze.

This tool avoids that by:

* reading MIDI **in a background thread**
* placing messages into `handle.rx_queue`
* letting you retrieve messages from the queue using **non-blocking** methods

---

## MIDI Clock Basics (for Sync)

### Core messages

* **Clock tick:** `mido.Message("clock")`

  * **24 ticks per quarter note (PPQN = 24)**
* **Transport:**

  * `start` — start playback from the beginning
  * `stop` — stop playback
  * `continue` — resume playback
* (Optional for some workflows) `songpos` — song position pointer (units of 16th notes)

### Two sync modes

1. **Computer is master** → you send `clock` + `start/stop/continue` to Digitakt
2. **Digitakt is master** → you receive `clock` (and transport) and follow it in your app

---

## Quick Start: List Ports

```python
from midi_tool_nb_clock import MidiToolNB

tool = MidiToolNB()
ports = tool.list()
print("Inputs:", ports["inputs"])
print("Outputs:", ports["outputs"])
```

If the Digitakt doesn’t match by name (sometimes it shows up as generic “USB MIDI”), choose ports manually (see below).

---

## Open a Device

### Auto-match (Digitakt)

```python
from midi_tool_nb_clock import MidiToolNB

tool = MidiToolNB()
handle = tool.open_digitakt()

print("IN :", handle.input_name)
print("OUT:", handle.output_name)
```

### Auto-match any device by keywords

```python
handle = tool.open_device(keywords=["my device brand", "model"])
```

### Manual selection (recommended if multiple similar ports appear)

```python
ports = tool.list()
print(ports["inputs"])
print(ports["outputs"])

handle = tool.open_device(
    input_port="EXACT INPUT PORT NAME",
    output_port="EXACT OUTPUT PORT NAME",
)
```

---

## Non-Blocking Receive (RX)

### Drain queued messages (non-blocking)

```python
import time
from midi_tool_nb_clock import MidiToolNB

tool = MidiToolNB()
h = tool.open_digitakt()

try:
    while True:
        msgs = tool.get_messages(h)  # non-blocking
        for msg in msgs:
            print("RX:", msg)
        time.sleep(0.01)
finally:
    h.close()
```

---

## Send MIDI (Notes, CC, Raw Messages)

```python
import time
from midi_tool_nb_clock import MidiToolNB

tool = MidiToolNB()
h = tool.open_digitakt()

try:
    # Note on/off
    tool.send_note_on(h, note=60, velocity=100, channel=0)
    time.sleep(0.2)
    tool.send_note_off(h, note=60, channel=0)

    # CC
    tool.send_cc(h, control=1, value=64, channel=0)

    # Raw message
    import mido
    tool.send(h, mido.Message("program_change", program=10, channel=0))
finally:
    h.close()
```

---

## Synchronization Examples

### A) Computer is Master → Digitakt Follows (Send MIDI Clock + Start/Stop)

This is the typical “sync external hardware to your computer” setup.

```python
import time
from midi_tool_nb_clock import MidiToolNB

tool = MidiToolNB()
h = tool.open_digitakt()

print("Sending clock to:", h.output_name)

try:
    # Start transport + clock at 120 BPM
    tool.start_clock(h, bpm=120.0, send_start=True, high_precision=True)

    # Keep running for 10 seconds
    time.sleep(10)

    # Stop transport (and clock)
    tool.stop_clock(h, send_stop=True)
finally:
    h.close()
```

**Digitakt settings tip:** enable receiving **clock/transport** from the relevant MIDI source (often “USB MIDI”).
(Exact menu names can vary by firmware.)

#### Notes on timing quality

* Python timing has jitter, especially under CPU load.
* `high_precision=True` uses `time.perf_counter()` scheduling and is usually better than `time.sleep()` alone.
* If you need “DAW-tight” sync, it’s often best to let the DAW generate clock and use this tool for scripting/monitoring.

---

### B) Digitakt is Master → Computer Follows (Receive Clock + Estimate BPM)

If Digitakt is sending clock, your script can measure it and adapt.

```python
import time
from midi_tool_nb_clock import MidiToolNB, ClockTracker

tool = MidiToolNB()
h = tool.open_digitakt()

tracker = ClockTracker(smoothing=0.15)
print("Receiving clock from:", h.input_name)

try:
    while True:
        msgs = tool.get_messages(h)
        for msg in msgs:
            bpm = tracker.update(msg)
            if bpm is not None:
                print(f"Estimated BPM: {bpm:.2f}")
        time.sleep(0.01)
finally:
    h.close()
```

#### What this gives you

* An approximate BPM estimate derived from MIDI clock intervals
* A way to “lock” your app’s tempo to Digitakt’s tempo source

---

### C) Follow Transport (Start/Stop/Continue) from Digitakt

```python
import time
from midi_tool_nb_clock import MidiToolNB

tool = MidiToolNB()
h = tool.open_digitakt()

try:
    while True:
        for msg in tool.get_messages(h):
            if msg.type in ("start", "stop", "continue"):
                print("Transport:", msg.type)
        time.sleep(0.01)
finally:
    h.close()
```

---

## Latency vs CPU Tuning

Two knobs matter:

### RX polling sleep

When no MIDI is available, the RX thread sleeps `poll_sleep`.

* `poll_sleep=0.001` (1ms): low latency, low CPU (good default)
* `0.005`–`0.01`: less CPU, slightly higher latency

Set in `open_device(..., poll_sleep=...)`.

### Clock scheduling

* `high_precision=True`: schedules ticks based on `perf_counter()` (recommended)
* `high_precision=False`: simpler `sleep(interval)` loop (more jitter)

Set in `start_clock(..., high_precision=...)`.

---

## Queue Overflow Policy

Incoming messages are put into a bounded queue (`queue_maxsize`, default 4096).

Current policy:

* If the queue is full, **new messages are dropped** (to avoid blocking the RX thread)

If you prefer a different policy (drop oldest, or block briefly), modify `_rx_loop`.

---

## Clean Shutdown

Always close the handle:

```python
h.close()
```

This stops:

* clock thread (if running)
* RX listener thread
* closes MIDI input/output ports

---

## Troubleshooting

### No MIDI ports appear

* Ensure backend is installed:

  ```bash
  pip install python-rtmidi
  ```
* Try replugging device / different cable / powered hub

### Digitakt not found by keyword match

* It might show as “USB MIDI Device” or similar.
* Use `tool.list()` and pass explicit `input_port`/`output_port`.

### Multiple ports for the same device

* Hardware may expose several ports (e.g., “MIDI 1”, “MIDI 2”).
* Try each output port and see which one actually drives the Digitakt,
  then hardcode that port via `output_port=...`.

---

## Practical Advice for Syncing Digitakt to a Computer

* If you’re already using a DAW (Ableton/Bitwig/Logic), it typically produces better MIDI clock than a Python script.
* Use this tool when:

  * you’re building a custom Python sequencer / controller
  * you want to monitor incoming clock/transport
  * you want to send clock to hardware in a lightweight script
* For best stability:

  * minimize CPU load
  * avoid running clock generation in the same thread as heavy work
  * prefer `high_precision=True`

---

