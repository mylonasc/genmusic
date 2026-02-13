import time
import fluidsynth

# 1. Initialize the synthesizer
fs = fluidsynth.Synth()

# 2. Start the audio driver (alsa, pulseaudio, etc.)
# On Ubuntu, 'alsa' or 'pulseaudio' usually works best
fs.start(driver='pulseaudio') 

# 3. Load your SoundFont
sfid = fs.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")
fs.program_select(0, sfid, 0, 0)

# 4. Play a note (Channel, Key, Velocity)
# Key 60 is Middle C
fs.noteon(0, 60, 100)

# 5. Hold the note for 2 seconds
time.sleep(2.0)

# 6. Stop the note and clean up
fs.noteoff(0, 60)
time.sleep(1.0)
fs.delete()