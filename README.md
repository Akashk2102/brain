# MindPrint - Assistive Technology Platform

## 🧠 Complete Brain-Computer Interface System

### Overview
MindPrint is a comprehensive assistive technology platform that combines eye-tracking, EEG analysis, machine learning, and AI assistance to help individuals with special needs interact with technology.

## 📁 Project Structure

### Core Files
- `unified_server.py` - **MAIN SERVER** - Integrated Flask server with all brainwave features
- `index.html` - Main website interface with integrated dashboard
- `brw.py` - Brainwave prediction API (standalone)
- `eg.py` - Eye tracking server with camera integration
- `sa.py` - Smart assistant API with Gemini Pro integration

### Legacy Files (Integrated into unified_server.py)
- `simulates_brainwaves.py` - EEG simulation (now integrated)
- `mindprint_dashboard.py` - Streamlit dashboard (now integrated)
- `test_client.py` - WebSocket test client (now integrated)

### Data Files
- `thoughts.csv` - Training data for ML model
- `training_data.csv` - Generated training data
- `user_data.db` - User database

## 🚀 Quick Start

### Option 1: Unified System (Recommended)
```bash
cd brain
python unified_server.py
```
Access at: http://localhost:5000

### Option 2: Individual Services
```bash
# Terminal 1: Eye Tracking
python eg.py

# Terminal 2: Brainwave API
python brw.py

# Terminal 3: Smart Assistant
python sa.py

# Terminal 4: EEG Simulation (legacy)
python simulates_brainwaves.py

# Terminal 5: Dashboard (legacy)
streamlit run mindprint_dashboard.py
```

## 🌐 Access Points

### Unified System
- **Main Interface**: http://localhost:5000
- **Integrated Dashboard**: Built into main website
- **All Features**: Brainwave simulation, ML predictions, training data

### Individual Services
- **Eye Tracking**: http://localhost:5000 (eg.py)
- **Brainwave API**: http://localhost:5001 (brw.py)
- **Smart Assistant**: http://localhost:5002 (sa.py)
- **EEG Simulation**: ws://localhost:8765 (simulates_brainwaves.py)
- **Dashboard**: http://localhost:8501 (mindprint_dashboard.py)

## 🎯 Features

### Integrated Brainwave Dashboard
- ✅ Real-time EEG simulation
- ✅ Live brainwave visualization (Delta, Theta, Alpha, Beta, Gamma)
- ✅ ML-based thought prediction
- ✅ Training data generation and management
- ✅ Model reloading and updates
- ✅ Real-time prediction testing

### Eye Tracking
- ✅ Camera-based gaze detection
- ✅ Quadrant-based selection system
- ✅ Real-time video feed
- ✅ Blink-based selection
- ✅ Calibration system

### Smart Assistant
- ✅ Gemini Pro AI integration
- ✅ Voice input/output
- ✅ Text chat interface
- ✅ Speech synthesis

### Machine Learning
- ✅ SVM-based thought classification
- ✅ Real-time brainwave prediction
- ✅ Confidence scoring
- ✅ Training data management
- ✅ Model retraining

## 🔧 Technical Details

### Dependencies
```bash
pip install flask flask-cors pandas scikit-learn scipy numpy websockets streamlit altair nest-asyncio streamlit-autorefresh opencv-python mediapipe requests
```

### Environment Variables
- `GEMINI_API_KEY` - Required for smart assistant functionality

### Architecture
- **Frontend**: HTML/CSS/JavaScript with integrated dashboard
- **Backend**: Flask with background threads for real-time processing
- **ML**: Scikit-learn SVM with StandardScaler pipeline
- **Real-time**: WebSocket and polling-based data updates

## 📊 API Endpoints (Unified Server)

### Brainwave Simulation
- `GET /api/brainwave-data` - Get latest brainwave data
- `GET /api/brainwave-history` - Get brainwave data history
- `POST /api/set-brainwave-state` - Set simulation state

### ML Predictions
- `POST /api/predict-thought` - Predict thought from brainwave data
- `POST /api/eeg-simulate` - Simulate EEG from keyboard input
- `POST /api/reload-model` - Reload ML model

### Training Data
- `POST /api/generate-training-data` - Generate new training dataset

### Dashboard
- `GET /api/dashboard-data` - Get data for visualization

## 🎮 Usage Instructions

### 1. Start the System
```bash
cd brain
python unified_server.py
```

### 2. Access the Interface
- Open browser to http://localhost:5000
- Navigate to "Brain Dashboard" section

### 3. Use Brainwave Features
- Click "Start Brainwave Simulation"
- Select brainwave state from dropdown
- View real-time brainwave data and charts
- Generate training data and test predictions

### 4. Use Eye Tracking
- Click "Start Eye Tracking Demo"
- Allow camera access
- Look at different quadrants to select options
- Close eyes for 3 seconds to confirm selection

### 5. Use Smart Assistant
- Click "Activate AI Assistant"
- Type messages or use voice input
- Get AI responses with speech output

## 🔄 Integration Benefits

The unified system eliminates the need to run multiple services separately:
- ❌ No more separate WebSocket server
- ❌ No more separate Streamlit dashboard
- ❌ No more separate test client
- ✅ Everything integrated into one server
- ✅ Single access point
- ✅ Simplified deployment
- ✅ Better performance

## 🛠️ Troubleshooting

### Common Issues
1. **Port conflicts**: Make sure ports 5000, 5001, 5002, 8501, 8765 are available
2. **Missing dependencies**: Install all required packages
3. **Camera access**: Allow browser camera access for eye tracking
4. **API keys**: Set GEMINI_API_KEY for smart assistant

### File Locations
All files are located in the `brain/` directory:
- Core Python files in `brain/`
- HTML interface: `brain/index.html`
- Data files: `brain/thoughts.csv`, `brain/training_data.csv`

## 📝 Notes

- The unified server combines all brainwave-related functionality
- Individual services can still be run separately if needed
- The HTML interface includes all dashboard features
- Real-time updates use polling instead of WebSockets for simplicity
- All brainwave simulation runs in background threads

## 🎉 Success!

Your complete MindPrint assistive technology platform is ready to use with all features integrated into a single, cohesive system!
