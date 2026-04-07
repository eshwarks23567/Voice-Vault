# Test Audio Directory

This directory contains sample audio files for testing the voice inventory system.

## Required Test Files

Create or download the following audio files:

### 1. `sample.wav`
Clear speech recording saying:
"Product ABC-123 quantity 50 location A-12"

### 2. `noisy_sample.wav`
Same phrase with background noise (factory sounds, machinery)

### 3. `fast_speech.wav`
Same phrase spoken quickly

### 4. `slow_speech.wav`
Same phrase spoken slowly with pauses

### 5. `accented_speech.wav`
Same phrase with different accent

## Creating Test Audio

### Using Python (with librosa)

```python
import numpy as np
import soundfile as sf

# Create a 3-second placeholder audio file
sample_rate = 16000
duration = 3.0
samples = int(sample_rate * duration)

# Generate silence (placeholder)
audio = np.zeros(samples, dtype=np.float32)

# Save as WAV
sf.write('sample.wav', audio, sample_rate)
```

### Using FFmpeg

```bash
# Generate 3 seconds of silence
ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 3 sample.wav

# Convert existing audio
ffmpeg -i input.mp3 -ar 16000 -ac 1 sample.wav
```

## Downloading Noise Samples

For background noise testing:
- Visit https://freesound.org
- Search for "factory noise", "warehouse sounds", "machinery"
- Download and mix with speech samples

## Audio Requirements

- Format: WAV (preferred), MP3, FLAC
- Sample rate: 16000 Hz (recommended) or higher
- Channels: Mono (stereo will be converted)
- Duration: 0.5 - 60 seconds
