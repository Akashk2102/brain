import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import time
import json
import asyncio
import websockets

# Function to simulate raw brain-wave data
def simulate_eeg(sample_rate=256, duration=1, state='relaxed'):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq_bands = {
        'delta': (1, 4, 0.5 if state == 'bored' else 0.1),
        'theta': (4, 8, 0.3 if state == 'depressed' else 0.2),
        'alpha': (8, 12, 0.8 if state == 'thirsty' else 0.4),
        'beta': (12, 30, 0.6 if state == 'hungry' else 0.3),
        'gamma': (30, 45, 0.2 if state == 'emergency' else 0.1)
    }
    
    raw = np.zeros_like(t)
    for band, (low, high, amp) in freq_bands.items():
        freq = np.random.uniform(low, high)
        phase = np.random.uniform(0, 2 * np.pi)
        raw += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # Add noise and filter for realism
    noise = np.random.normal(0, 0.1, len(t))
    raw += noise
    b, a = signal.butter(4, [1, 45], btype='band', fs=sample_rate)
    raw = signal.filtfilt(b, a, raw)
    
    return raw, time.time()

# Function to compute band powers from raw data
def compute_band_powers(raw, fs=256):
    N = len(raw)
    yf = fft(raw)
    xf = fftfreq(N, 1 / fs)[:N//2]
    powers = 2.0 / N * np.abs(yf[:N//2])
    
    bands = {
        'delta': np.mean(powers[(xf >= 1) & (xf < 4)]),
        'theta': np.mean(powers[(xf >= 4) & (xf < 8)]),
        'alpha': np.mean(powers[(xf >= 8) & (xf < 12)]),
        'beta': np.mean(powers[(xf >= 12) & (xf < 30)]),
        'gamma': np.mean(powers[(xf >= 30) & (xf < 45)])
    }
    return bands

# WebSocket server for real-time streaming
async def handler(websocket):
    while True:
        raw, ts = simulate_eeg(state='relaxed')  # Change 'relaxed' to other states as needed
        bands = compute_band_powers(raw)
        payload = {'timestamp': ts, **bands}
        await websocket.send(json.dumps(payload))
        await asyncio.sleep(1)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()  # Keeps the server running forever

if __name__ == "__main__":
    asyncio.run(main())
