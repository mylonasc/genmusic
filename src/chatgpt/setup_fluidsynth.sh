#!/bin/bash
# FluidSynth Setup Script for Linux
# Run: ./setup_fluidsynth.sh

set -e

echo "=== FluidSynth Setup for genmusic ==="

# Check for pip
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    sudo apt-get update && sudo apt-get install -y python3-pip
fi

# Install pyfluidsynth (using python3 -m pip to match system python)
echo "Installing pyfluidsynth..."
python3 -m pip install --user pyfluidsynth

# Check for soundfonts
echo ""
echo "=== Available SoundFonts ==="
if [ -d "/usr/share/sounds/sf2" ]; then
    ls -la /usr/share/sounds/sf2/
else
    echo "Warning: /usr/share/sounds/sf2 not found"
    echo "You may need to install: sudo apt-get install fluidsynth soundfont-fluid"
fi

# Check for audio drivers
echo ""
echo "=== Available Audio Drivers ==="
fluidsynth -a help 2>&1 | grep -E "^\s+-a|^   '\w+'" || true

echo ""
echo "=== Audio Devices ==="
echo "ALSA playback devices:"
aplay -l 2>&1 | grep -E "^card" || echo "  None found"

echo ""
echo "PulseAudio sinks:"
pactl list short sinks 2>&1 | grep -E "sink" || echo "  None found (PulseAudio may not be running)"

# Test audio
echo ""
echo "=== Testing FluidSynth ==="
python3 -c "
import fluidsynth
import time
fs = fluidsynth.Synth()
fs.start(driver='alsa')
sfid = fs.sfload('/usr/share/sounds/sf2/FluidR3_GM.sf2')
fs.program_select(0, sfid, 0, 0)
fs.noteon(0, 60, 100)
time.sleep(0.2)
fs.noteoff(0, 60)
time.sleep(0.1)
fs.delete()
print('FluidSynth audio test: OK')
" 2>&1 | grep -v "warning:" || true

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the demo:"
echo "  cd src/chatgpt"
echo "  python3 -u main_fluidsynth.py"
echo ""
echo "If you need a different audio driver, edit main_fluidsynth.py:"
echo "  driver = 'alsa'       # for direct hardware access"
echo "  driver = 'pulseaudio' # for PulseAudio"
echo "  driver = 'jack'       # for JACK audio"
echo ""
echo "Environment variables (optional):"
echo "  FLUIDSYNTH_SOUNDFONT=/path/to/soundfont.sf2"
echo "  FLUIDSYNTH_DRIVER=alsa"
