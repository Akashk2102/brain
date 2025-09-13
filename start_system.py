#!/usr/bin/env python3
"""
MindPrint System Startup Script
===============================

This script provides an easy way to start the MindPrint assistive technology platform.
It handles all the setup and provides options for different deployment modes.

Usage:
    python start_system.py [option]

Options:
    unified    - Start unified server (recommended)
    individual - Start all services individually
    eye-only   - Start only eye tracking
    brain-only - Start only brainwave services
    help       - Show this help message
"""

import sys
import subprocess
import time
import os
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("🧠 MindPrint - Assistive Technology Platform")
    print("=" * 60)
    print("Complete Brain-Computer Interface System")
    print("Eye Tracking | EEG Analysis | AI Assistance")
    print("=" * 60)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'flask_cors', 'pandas', 'sklearn', 'scipy', 
        'numpy', 'websockets', 'streamlit', 'altair', 'cv2', 'mediapipe'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("pip install flask flask-cors pandas scikit-learn scipy numpy websockets streamlit altair opencv-python mediapipe")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def start_unified_server():
    """Start the unified server with all integrated features"""
    print("🚀 Starting Unified MindPrint Server...")
    print("📊 Brainwave simulation: ✅ Integrated")
    print("🧠 ML Model: ✅ Loaded")
    print("🌐 Web Interface: ✅ Ready")
    print("🔗 All services integrated!")
    print("\n🌍 Access your system at: http://localhost:5000")
    print("📱 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "unified_server.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 MindPrint server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting unified server: {e}")
    except FileNotFoundError:
        print("❌ unified_server.py not found. Make sure you're in the brain directory.")

def start_individual_services():
    """Start all services individually in separate processes"""
    print("🚀 Starting Individual MindPrint Services...")
    
    services = [
        ("Eye Tracking Server", "eg.py", 5000),
        ("Brainwave API", "brw.py", 5001),
        ("Smart Assistant", "sa.py", 5002),
        ("EEG Simulation", "simulates_brainwaves.py", 8765),
        ("Dashboard", "streamlit run mindprint_dashboard.py", 8501)
    ]
    
    processes = []
    
    try:
        for name, script, port in services:
            print(f"🔄 Starting {name} on port {port}...")
            
            if "streamlit" in script:
                process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "mindprint_dashboard.py"])
            else:
                process = subprocess.Popen([sys.executable, script])
            
            processes.append((name, process))
            time.sleep(2)  # Give each service time to start
        
        print("\n✅ All services started!")
        print("🌍 Access points:")
        print("   - Main Website: http://localhost:5000")
        print("   - Eye Tracking: http://localhost:5000")
        print("   - Brainwave API: http://localhost:5001")
        print("   - Smart Assistant: http://localhost:5002")
        print("   - Dashboard: http://localhost:8501")
        print("   - WebSocket: ws://localhost:8765")
        print("\n📱 Press Ctrl+C to stop all services")
        
        # Wait for interrupt
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        for name, process in processes:
            print(f"🔄 Stopping {name}...")
            process.terminate()
            process.wait()
        print("👋 All services stopped.")

def start_eye_tracking_only():
    """Start only the eye tracking service"""
    print("👁️ Starting Eye Tracking Service...")
    try:
        subprocess.run([sys.executable, "eg.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Eye tracking stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting eye tracking: {e}")

def start_brainwave_only():
    """Start only brainwave-related services"""
    print("🧠 Starting Brainwave Services...")
    
    services = [
        ("Brainwave API", "brw.py"),
        ("EEG Simulation", "simulates_brainwaves.py")
    ]
    
    processes = []
    
    try:
        for name, script in services:
            print(f"🔄 Starting {name}...")
            process = subprocess.Popen([sys.executable, script])
            processes.append((name, process))
            time.sleep(2)
        
        print("\n✅ Brainwave services started!")
        print("🌍 Access points:")
        print("   - Brainwave API: http://localhost:5001")
        print("   - WebSocket: ws://localhost:8765")
        print("\n📱 Press Ctrl+C to stop services")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping brainwave services...")
        for name, process in processes:
            print(f"🔄 Stopping {name}...")
            process.terminate()
            process.wait()
        print("👋 Brainwave services stopped.")

def show_help():
    """Show help information"""
    print(__doc__)
    print("\n📁 File Structure:")
    print("   brain/")
    print("   ├── unified_server.py     # Main unified server")
    print("   ├── index.html           # Web interface")
    print("   ├── eg.py                # Eye tracking")
    print("   ├── brw.py               # Brainwave API")
    print("   ├── sa.py                # Smart assistant")
    print("   ├── simulates_brainwaves.py  # EEG simulation")
    print("   ├── mindprint_dashboard.py   # Dashboard")
    print("   └── README.md            # Documentation")

def main():
    """Main function"""
    print_banner()
    
    # Check if we're in the right directory
    if not os.path.exists("unified_server.py"):
        print("❌ Error: unified_server.py not found!")
        print("💡 Make sure you're running this script from the brain/ directory")
        print("   cd brain")
        print("   python start_system.py")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        option = "unified"
    else:
        option = sys.argv[1].lower()
    
    # Execute based on option
    if option == "unified":
        start_unified_server()
    elif option == "individual":
        start_individual_services()
    elif option == "eye-only":
        start_eye_tracking_only()
    elif option == "brain-only":
        start_brainwave_only()
    elif option in ["help", "-h", "--help"]:
        show_help()
    else:
        print(f"❌ Unknown option: {option}")
        print("💡 Use 'python start_system.py help' for usage information")

if __name__ == "__main__":
    main()
