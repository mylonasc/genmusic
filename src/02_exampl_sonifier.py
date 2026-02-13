import random
import time
from scamp import Session

# --- 1. SIGNAL PROCESSING MODULE ---
class SignalProcessor:
    """Handles the transformation of raw data into clean, musical control signals."""
    def __init__(self, smoothing=1.):
        self.smoothing = smoothing
        self.last_val = 0.1

    def normalize(self, val, min_in, max_in):
        """Clamps and scales input to a 0.0 - 1.0 range."""
        if max_in == min_in: return 0.5
        norm = (val - min_in) / (max_in - min_in)
        return max(0.0, min(1.0, norm))

    def smooth(self, val):
        """Leaky integrator to prevent erratic note jumps."""
        smoothed = (self.smoothing * val) + (1 - self.smoothing) * self.last_val
        self.last_val = smoothed
        return smoothed

# --- 2. HARMONIC MODULE ---
class Harmonizer:
    """Manages scales, progressions, and pitch quantization."""
    def __init__(self):
        self.library = {
            "zen": [60, 62, 64, 67, 69],          # Major Pentatonic
            "air": [60, 62, 66, 67, 69, 72],      # Lydian
            "dark": [60, 61, 64, 65, 67, 68, 70], # Phrygian Dominant
            "mystery": [60, 63, 65, 66, 67, 70]   # Blues/Minor
        }
        self.current_mood = "zen"
        self.root_offset = 0 # Can be used to transpose the whole song

    def get_pitch(self, normalized_val):
        """Maps a 0-1 value to a specific note in the current scale."""
        scale = self.library[self.current_mood]
        idx = int(normalized_val * (len(scale) - 1))
        return scale[idx] + self.root_offset

    def change_mood(self, mood_name=None):
        if mood_name in self.library:
            self.current_mood = mood_name
        else:
            self.current_mood = random.choice(list(self.library.keys()))
        print(f"-> Harmony shifted to: {self.current_mood}")

# --- 3. THE CORE ENGINE ---
class GenerativeEngine:
    """The main coordinator that connects the signal to the instrument."""
    def __init__(self, instrument_name="Synth"):
        self.session = Session()
        self.instrument = self.session.new_part(instrument_name)
        self.processor = SignalProcessor(smoothing=1)
        self.harmonizer = Harmonizer()
        self.note_count = 0

    def process_and_play(self, raw_data, min_val=0, max_val=1024):
        # 1. Process Signal
        norm = self.processor.normalize(raw_data, min_val, max_val)
        smooth_val = self.processor.smooth(norm)
        
        # 2. Update Harmony (Change scale every 16 notes)
        self.note_count += 1
        if self.note_count % 32 == 0:
            self.harmonizer.change_mood()

        # 3. Calculate Musical Parameters
        pitch = self.harmonizer.get_pitch(smooth_val)
        
        # Advanced mapping: Higher signal = Louder & Faster
        volume = 0.4 + (smooth_val * 1) 
        duration = 0.2# - (smooth_val * 0.2) # 1.0s to 0.2s
        
        # 4. Play via SCAMP
        # We use a slight wait to ensure notes don't stack infinitely
        self.instrument.play_note(pitch, volume, duration)

def read_data(fname = 'VTV_data', smoothing_window = 30):

    import pandas as pd
    import numpy as np
    vtv = pd.read_pickle(fname)
    v = (vtv['Close'] - vtv['Close'].rolling(window=smoothing_window, min_periods=1).mean())
    vs = v.rolling(window = smoothing_window).std()
    vv = (v/vs).values
    vsig = vv[~np.isnan(vv)]
    vsig = (vsig-np.min(vsig))/(np.max(vsig)  - np.min(vsig))
    return vsig

# --- 4. EXECUTION ---
if __name__ == "__main__":
    # Setup
    dat = read_data()
    engine = GenerativeEngine("Electric Piano")
    
    print("Starting Generative Stream... Press Ctrl+C to stop.")
    
    icurr = 0
    try:
        while True:
            # SIMULATION: Replace random.randint with your actual signal (e.g., mushroom sensor)
            # raw_signal = random.randint(0, 1024)
            raw_signal = dat[icurr]*1024
            icurr += 1
            
            # This will block the loop for the length of the 'duration' 
            # produced inside the engine.
            engine.process_and_play(raw_signal)
            
    except KeyboardInterrupt:
        print("\nStopping Performance.")