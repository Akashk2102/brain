from flask import Flask, request, jsonify, Response, render_template_string
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import os
import threading
import time
import json
import asyncio
import websockets
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import random
import datetime
from collections import deque

app = Flask(__name__)
CORS(app)

# ==================== BRAINWAVE SIMULATION ====================
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

# Global variables for brainwave simulation
current_brainwave_state = 'relaxed'
brainwave_data_queue = deque(maxlen=60)
brainwave_thread = None

def brainwave_simulation_loop():
    global current_brainwave_state, brainwave_data_queue
    while True:
        try:
            raw, ts = simulate_eeg(state=current_brainwave_state)
            bands = compute_band_powers(raw)
            
            data = {
                'timestamp': ts,
                'delta': float(bands['delta']),
                'theta': float(bands['theta']),
                'alpha': float(bands['alpha']),
                'beta': float(bands['beta']),
                'gamma': float(bands['gamma']),
                'state': current_brainwave_state
            }
            
            brainwave_data_queue.append(data)
            time.sleep(1)
        except Exception as e:
            print(f"Brainwave simulation error: {e}")
            time.sleep(1)

# ==================== ML MODEL ====================
def load_model():
    """Load or reload the machine learning model from training data"""
    global clf, data, X, y, fixed_key_brainwaves
    
    try:
        if os.path.exists("training_data.csv"):
            data = pd.read_csv("training_data.csv")
            X = data.drop(columns=["thought"]).values
            y = data["thought"].values 
            
            clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
            clf.fit(X, y)
            
            fixed_key_brainwaves = {
                'a': X[y == 'thirsty'][0] if (y == 'thirsty').any() else X[0],
                's': X[y == 'hungry'][0] if (y == 'hungry').any() else X[0],
                'd': X[y == 'emergency'][0] if (y == 'emergency').any() else X[0],
                'f': X[y == 'depressed'][0] if (y == 'depressed').any() else X[0],
                'g': X[y == 'bored'][0] if (y == 'bored').any() else X[0]
            }
            print("✅ Model loaded successfully from training_data.csv")
        else:
            data = pd.read_csv("thoughts.csv")
            X = data.drop(columns=["thought"]).values
            y = data["thought"].values 
            
            clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
            clf.fit(X, y)
            
            fixed_key_brainwaves = {
                'a': X[y == 'thirsty'][0],
                's': X[y == 'hungry'][0],
                'd': X[y == 'emergency'][0] if (y == 'emergency').any() else X[0],
                'f': X[y == 'depressed'][0] if (y == 'depressed').any() else X[0],
                'g': X[y == 'bored'][0] if (y == 'bored').any() else X[0]
            }
            print("✅ Model loaded from thoughts.csv (fallback)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import numpy as np
        X = np.random.rand(10, 8)
        y = np.array(['relaxed'] * 10)
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
        clf.fit(X, y)
        fixed_key_brainwaves = {'a': X[0], 's': X[0], 'd': X[0], 'f': X[0], 'g': X[0]}

# Initialize the model
load_model()
last_eeg_state = {"thought": None}

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template_string(open('index.html').read())

@app.route('/api/brainwave-data')
def get_brainwave_data():
    """Get latest brainwave data"""
    if brainwave_data_queue:
        return jsonify(list(brainwave_data_queue)[-1])
    return jsonify({"error": "No data available"})

@app.route('/api/brainwave-history')
def get_brainwave_history():
    """Get brainwave data history"""
    return jsonify(list(brainwave_data_queue))

@app.route('/api/set-brainwave-state', methods=['POST'])
def set_brainwave_state():
    """Set the brainwave simulation state"""
    global current_brainwave_state
    data = request.get_json()
    state = data.get('state', 'relaxed')
    current_brainwave_state = state
    return jsonify({"status": "State updated", "state": state})

@app.route('/api/eeg-simulate', methods=['POST'])
def eeg_simulate():
    data = request.get_json()
    key = data.get('key', '').lower()
    
    if key in fixed_key_brainwaves:
        brainwave = fixed_key_brainwaves[key]
        thought = clf.predict([brainwave])[0]
        last_eeg_state["thought"] = thought
        return jsonify({"thought": thought})
    return jsonify({"thought": None})

@app.route('/api/predict-thought', methods=['POST'])
def predict_thought():
    """Predict thought from brainwave data"""
    try:
        data = request.get_json()
        
        brainwave_data = [
            data.get('ch1', 0),
            data.get('ch2', 0), 
            data.get('ch3', 0),
            data.get('ch4', 0),
            data.get('ch5', 0),
            data.get('ch6', 0),
            data.get('ch7', 0),
            data.get('ch8', 0)
        ]
        
        prediction = clf.predict([brainwave_data])[0]
        confidence = clf.predict_proba([brainwave_data])[0]
        
        classes = clf.classes_
        predicted_idx = list(classes).index(prediction)
        confidence_score = confidence[predicted_idx]
        
        last_eeg_state["thought"] = prediction
        
        return jsonify({
            "thought": prediction,
            "confidence": float(confidence_score),
            "all_probabilities": dict(zip(classes, confidence.tolist()))
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/eeg-latest', methods=['GET'])
def eeg_latest():
    return jsonify(last_eeg_state)

@app.route('/api/reload-model', methods=['POST'])
def reload_model():
    """Reload the model with updated training data"""
    try:
        load_model()
        return jsonify({"status": "Model reloaded successfully", "data_shape": X.shape})
    except Exception as e:
        return jsonify({"status": "Error reloading model", "error": str(e)}), 500

@app.route('/api/generate-training-data', methods=['POST'])
def generate_training_data():
    """Generate training data for ML model"""
    try:
        data = request.get_json()
        n_samples = data.get('n_samples', 100)
        
        thoughts = ['hungry', 'thirsty', 'emergency', 'depressed', 'bored', 'relaxed']
        rows = []
        
        for _ in range(n_samples):
            thought = random.choice(thoughts)
            rows.append({
                "thought": thought,
                "ch1": round(random.uniform(0, 1), 3),
                "ch2": round(random.uniform(0, 1), 3),
                "ch3": round(random.uniform(0, 1), 3),
                "ch4": round(random.uniform(0, 1), 3),
                "ch5": round(random.uniform(0, 1), 3),
                "ch6": round(random.uniform(0, 1), 3),
                "ch7": round(random.uniform(0, 1), 3),
                "ch8": round(random.uniform(0, 1), 3),
            })
        
        df = pd.DataFrame(rows)
        df.to_csv("training_data.csv", index=False)
        
        return jsonify({
            "status": "Training data generated successfully",
            "samples": len(rows),
            "preview": rows[:5]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard-data')
def get_dashboard_data():
    """Get data for dashboard visualization"""
    if brainwave_data_queue:
        return jsonify(list(brainwave_data_queue))
    return jsonify([])

if __name__ == '__main__':
    # Start brainwave simulation in background thread
    brainwave_thread = threading.Thread(target=brainwave_simulation_loop, daemon=True)
    brainwave_thread.start()
    
    print("🚀 Starting Unified MindPrint Server...")
    print("📊 Brainwave simulation: ✅ Running")
    print("🧠 ML Model: ✅ Loaded")
    print("🌐 Web Interface: ✅ Ready")
    print("🔗 All services integrated!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

