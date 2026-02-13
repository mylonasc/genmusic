import time

class BioSonifier:
    def __init__(self, scale=None, smoothing=0.15):
        """
        :param scale: List of MIDI notes (e.g., [60, 62, 64, 67, 69])
        :param smoothing: 0.0 to 1.0 (Leaky integrator coefficient)
        """
        # Default: C Major Pentatonic (Always sounds good)
        self.scale = scale if scale else [60, 62, 64, 67, 69, 72, 74, 76, 79]
        self.smoothing = smoothing
        self.last_val = 0.5
        
    def _leaky_integrator(self, new_val):
        """Prevents 'teleporting' notes by smoothing the signal."""
        smoothed = (self.smoothing * new_val) + (1 - self.smoothing) * self.last_val
        self.last_val = smoothed
        return smoothed

    def _quantize(self, value):
        """Maps a 0.0-1.0 value to the nearest note in the scale."""
        idx = int(value * (len(self.scale) - 1))
        return self.scale[idx]

    def process(self, raw_signal, min_in=0, max_in=1024):
        """
        The main engine.
        :param raw_signal: The raw float/int from your mushroom/sensor
        :return: A dict of musical parameters
        """
        # 1. Normalize to 0.0 - 1.0
        norm_val = (raw_signal - min_in) / (max_in - min_in)
        norm_val = max(0, min(1, norm_val)) # Clamp
        
        # 2. Smooth the signal
        smooth_val = self._leaky_integrator(norm_val)
        
        # 3. Derive Musicality
        return {
            "note": self._quantize(smooth_val),
            "velocity": int(60 + (smooth_val * 40)), # Dynamics: Louder when signal is higher
            "duration": 0.1 + (1 - smooth_val) * 0.5, # Faster notes for higher signals
            "cutoff": int(smooth_val * 127) # For filter modulation
        }
    
from scamp import Session
import random # Replace this with your mushroom signal input
import time

# s = Session()
s = Session(default_audio_driver='pulseaudio')
piano = s.new_part("piano")

# Define a Pentatonic Scale (MIDI numbers)
scale = [60, 62, 64, 67, 69, 72, 74, 76, 79, 81]
drop_prob = 0.7
duration = 0.5

while True:
    # 1. Get raw signal (0.0 to 1.0)
    raw_val = random.random() 
    v = random.random()
    print(v)
    # 2. Map to scale index
    note_index = int(raw_val * (len(scale) - 1))
    pitch = scale[note_index]
    if v > drop_prob:
        time.sleep(duration)
        continue
    
    # 3. Play note (pitch, volume, duration)
    piano.play_note(pitch, 0.3, duration)
