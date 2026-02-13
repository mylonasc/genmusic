import random
from scamp import Session

# --- Rhythmic Helper ---
def generate_euclidean(pulses, steps):
    if pulses > steps: pulses = steps
    pattern = [False] * steps
    if pulses == 0: return pattern
    for i in range(pulses):
        pattern[(i * steps) // pulses] = True
    return pattern


# --- THE DUAL ENGINE ---
class DualLayerEngine:
    def __init__(self):
        self.session = Session()
        self.session.tempo = 120
        
        # 1. Instruments
        self.lead = self.session.new_part("Marimba")
        self.bass = self.session.new_part("Piano")
        
        # 2. Harmonic State
        self.scale = [48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72] # Extended C Pentatonic
        self.lead_idx = len(self.scale) // 2
        self.bass_idx = 2 # Start bass low
        
        # 3. Rhythmic State
        self.steps_per_loop = 64
        self.lead_rhythm = generate_euclidean(7, 64) # Busy Lead
        self.bass_rhythm = generate_euclidean(3, 64) # Sparse Bass
        self.current_step = 0

    def get_walk_step(self, signal_val):
        """Translates signal intensity into a directional step."""
        if signal_val > 0.7: return 1   # Go Up
        if signal_val < 0.3: return -1  # Go Down
        return random.choice([-1, 0, 1]) # Drift

    def play_tick(self, signal_val):
        """Processes one clock tick for both layers."""
        
        # --- LEAD LAYER (Euclidean Arp) ---
        if self.lead_rhythm[self.current_step]:
            # Lead follows the signal quickly
            self.lead_idx = (self.lead_idx + self.get_walk_step(signal_val)) % len(self.scale)
            # Constrain Lead to upper half of scale
            lead_pitch = self.scale[max(4, self.lead_idx)]
            self.lead.play_note(lead_pitch, 0.6 + (signal_val * 0.3), 0.15, blocking=False)

        # --- BASS LAYER (Drone/Foundation) ---
        if self.bass_rhythm[self.current_step]:
            # Bass moves more slowly/conservatively
            if random.random() > 0.5: # Only change bass half the time
                self.bass_idx = (self.bass_idx + self.get_walk_step(signal_val)) % 5 
            
            bass_pitch = self.scale[self.bass_idx]
            # Bass notes are longer and softer
            self.bass.play_note(bass_pitch, 0.5, 0.6, blocking=False)

        # --- CLOCK MANAGEMENT ---
        self.current_step = (self.current_step + 1) % self.steps_per_loop
        # Wait for a 16th note duration
        self.session.wait(0.125) 

# --- EXECUTION LOOP ---
if __name__ == "__main__":
    engine = DualLayerEngine()
    
    # Example: A signal that fluctuates (like stock data or a sensor)
    # In a real app, you'd pull your scaled 0.0-1.0 value here
    try:
        while True:
            # Simulate a fluctuating signal
            simulated_signal = abs(random.uniform(0, 1)) 
            engine.play_tick(simulated_signal)
    except KeyboardInterrupt:
        print("Performance ended.")