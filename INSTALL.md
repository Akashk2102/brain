# MindPrint Installation Guide

## 🚀 Quick Installation

### 1. Prerequisites
- Python 3.8 or higher
- Webcam (for eye tracking)
- Internet connection (for AI features)

### 2. Install Dependencies
```bash
# Navigate to the brain directory
cd brain

# Install all required packages
pip install -r requirements.txt
```

### 3. Start the System
```bash
# Option 1: Unified server (recommended)
python start_system.py

# Option 2: Direct unified server
python unified_server.py

# Option 3: Individual services
python start_system.py individual
```

### 4. Access the Interface
- Open your browser
- Go to: http://localhost:5000
- Enjoy your complete MindPrint system!

## 🔧 Detailed Setup

### Environment Setup
```bash
# Create virtual environment (recommended)
python -m venv mindprint_env

# Activate virtual environment
# Windows:
mindprint_env\Scripts\activate
# macOS/Linux:
source mindprint_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: AI Assistant Setup
```bash
# Set environment variable for Gemini Pro AI
# Windows:
set GEMINI_API_KEY=your_api_key_here
# macOS/Linux:
export GEMINI_API_KEY=your_api_key_here
```

### Verify Installation
```bash
# Run the startup script to check dependencies
python start_system.py help
```

## 📁 File Structure After Installation

```
brain/
├── unified_server.py          # Main server (integrated everything)
├── start_system.py           # Easy startup script
├── index.html               # Web interface
├── brw.py                   # Brainwave API (standalone)
├── eg.py                    # Eye tracking server
├── sa.py                    # Smart assistant API
├── simulates_brainwaves.py  # EEG simulation (legacy)
├── mindprint_dashboard.py   # Dashboard (legacy)
├── test_client.py           # Test client (legacy)
├── thoughts.csv             # Training data
├── training_data.csv        # Generated training data
├── user_data.db             # User database
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── INSTALL.md              # This file
└── start_system.py         # Startup script
```

## 🎯 Usage Options

### Unified System (Recommended)
- Single server with all features
- Integrated dashboard
- Real-time brainwave simulation
- ML predictions
- Training data management

### Individual Services
- Separate processes for each feature
- More control over individual components
- Useful for development and debugging

## 🛠️ Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill processes using ports
   netstat -ano | findstr :5000
   taskkill /PID <PID_NUMBER> /F
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install --upgrade -r requirements.txt
   ```

3. **Camera Access Issues**
   - Allow browser camera permissions
   - Check if camera is being used by another application

4. **AI Assistant Not Working**
   - Verify GEMINI_API_KEY is set
   - Check internet connection

### Platform-Specific Notes

#### Windows
- Use Command Prompt or PowerShell
- May need to run as administrator for some operations

#### macOS
- May need to install Xcode command line tools
- Camera permissions may need manual approval

#### Linux
- May need additional packages for OpenCV
- Install with: `sudo apt-get install python3-opencv`

## 🎉 Success!

Once installation is complete, you'll have access to:

- ✅ **Real-time Brainwave Simulation**
- ✅ **Integrated Dashboard**
- ✅ **ML-based Thought Prediction**
- ✅ **Eye Tracking with Camera**
- ✅ **AI Smart Assistant**
- ✅ **Training Data Management**
- ✅ **Complete Web Interface**

Your MindPrint assistive technology platform is ready to help individuals with special needs interact with technology in new and innovative ways!
