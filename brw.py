from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)

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
            
            # Update fixed key brainwaves with new data
            fixed_key_brainwaves = {
                'a': X[y == 'thirsty'][0] if (y == 'thirsty').any() else X[0],
                's': X[y == 'hungry'][0] if (y == 'hungry').any() else X[0],
                'd': X[y == 'emergency'][0] if (y == 'emergency').any() else X[0],
                'f': X[y == 'depressed'][0] if (y == 'depressed').any() else X[0],
                'g': X[y == 'bored'][0] if (y == 'bored').any() else X[0]
            }
            print("✅ Model loaded successfully from training_data.csv")
        else:
            # Fallback to thoughts.csv if training_data.csv doesn't exist
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
        # Initialize with dummy data if all else fails
        import numpy as np
        X = np.random.rand(10, 8)
        y = np.array(['relaxed'] * 10)
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
        clf.fit(X, y)
        fixed_key_brainwaves = {'a': X[0], 's': X[0], 'd': X[0], 'f': X[0], 'g': X[0]}

# Initialize the model
load_model()

last_eeg_state = {"thought": None}

def predict_thought_from_key(letter):
    if letter not in fixed_key_brainwaves:
        return None
    brainwave = fixed_key_brainwaves[letter]
    return clf.predict([brainwave])[0]

@app.route('/api/eeg-simulate', methods=['POST'])
def eeg_simulate():
    data = request.get_json()
    key = data.get('key', '').lower()
    thought = predict_thought_from_key(key)
    last_eeg_state["thought"] = thought
    return jsonify({"thought": thought})

@app.route('/api/predict-thought', methods=['POST'])
def predict_thought():
    """Predict thought from brainwave data sent from dashboard"""
    try:
        data = request.get_json()
        
        # Extract brainwave channels
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
        
        # Predict thought using the ML model
        prediction = clf.predict([brainwave_data])[0]
        confidence = clf.predict_proba([brainwave_data])[0]
        
        # Get confidence for the predicted class
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

if __name__ == '__main__':
    app.run(debug=True, port=5001)