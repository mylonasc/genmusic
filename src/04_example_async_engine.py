import scamp
import asyncio
import random

# --- THE CRITICAL FIX ---
# This tells SCAMP to try the system's version of FluidSynth first, 
# which often bypasses the 'priority' buzz issue.
scamp.playback_settings.try_system_fluidsynth_first = True

from scamp import Session

class EventDispatcher:
    def __init__(self):
        self.listeners = {}
    def subscribe(self, event, callback):
        self.listeners.setdefault(event, []).append(callback)
    def emit(self, event, data=None):
        for callback in self.listeners.get(event, []): callback(data)

class EuclideanRhythm:
    def __init__(self, p, s):
        self.pattern = [(i * p) // s != ((i - 1) * p) // s for i in range(s)]
        self.idx = 0
    def next_step(self):
        val = self.pattern[self.idx]
        self.idx = (self.idx + 1) % len(self.pattern)
        return val

class GenerativeSystem:
    def __init__(self):
        self.session = Session()
        self.instr = self.session.new_part("Piano")
        self.rhythm = EuclideanRhythm(3, 8) # Sparser rhythm (3 beats) to stop buzzing
        self.signal = 0.5
        self.bus = EventDispatcher()
        self.bus.subscribe("shift", self.change_rhythm)

    def change_rhythm(self, _):
        self.rhythm = EuclideanRhythm(random.choice([2, 3, 5]), 8)
        print("Rhythm Shifted!")

    def music_loop(self):
        # We increase the wait time (0.2) to reduce CPU load
        tick = 0.2 
        scale = [60, 62, 64, 67, 69]
        while True:
            if self.rhythm.next_step():
                pitch = scale[int(self.signal * (len(scale)-1))]
                # Use a very short duration (0.1) so notes NEVER overlap
                self.instr.play_note(pitch, 0.5, 0.1, blocking=False)
            self.session.wait(tick)

    async def signal_watcher(self):
        while True:
            self.signal = random.random()
            if self.signal > 0.9: self.bus.emit("shift")
            await asyncio.sleep(0.5)

    def run(self):
        self.session.fork(self.music_loop)
        # Proper way to integrate asyncio with SCAMP's loop
        loop = asyncio.get_event_loop()
        loop.create_task(self.signal_watcher())
        self.session.wait_forever()

if __name__ == "__main__":
    GenerativeSystem().run()