#!/usr/bin/env python3
"""
List MIDI ports and try to detect an Elektron Digitakt.

Requires:
  pip install mido python-rtmidi
"""

from __future__ import annotations
import mido

KEYWORDS = [
    "digitakt",
    "elektron",
]

def score_port(name: str) -> int:
    n = name.lower()
    return sum(1 for k in KEYWORDS if k in n)

def main() -> None:
    in_ports = mido.get_input_names()
    out_ports = mido.get_output_names()

    print("=== MIDI INPUT PORTS ===")
    if not in_ports:
        print("  (none found)")
    for i, name in enumerate(in_ports):
        print(f"  [{i}] {name}")

    print("\n=== MIDI OUTPUT PORTS ===")
    if not out_ports:
        print("  (none found)")
    for i, name in enumerate(out_ports):
        print(f"  [{i}] {name}")

    # Try to detect Digitakt-like ports by keyword match
    detected_in = [p for p in in_ports if score_port(p) > 0]
    detected_out = [p for p in out_ports if score_port(p) > 0]

    print("\n=== DETECTION (keyword match) ===")
    if detected_in:
        print("Likely Digitakt INPUT port(s):")
        for p in detected_in:
            print(f"  - {p}")
    else:
        print("No obvious Digitakt INPUT ports found (by name).")

    if detected_out:
        print("Likely Digitakt OUTPUT port(s):")
        for p in detected_out:
            print(f"  - {p}")
    else:
        print("No obvious Digitakt OUTPUT ports found (by name).")

    print("\nTip: If you see multiple 'Elektron'/'Digitakt' ports, try each one for IO.")
    print("On some systems, ports may appear as 'Elektron Digitakt MIDI 1', 'MIDI 2', etc.")

if __name__ == "__main__":
    main()
