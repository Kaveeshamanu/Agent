#!/usr/bin/env python3
"""
SteamNoodles Feedback Agent - Quick Demo Launcher
Simplified launcher for the interactive demo
"""

import os
import sys
import subprocess

def main():
    """Launch the demo with proper setup"""
    print("🍜 SteamNoodles Feedback Agent - Quick Demo Launcher")
    print("=" * 55)
    
    # Check if we're in the right directory
    if not os.path.exists('demo.py'):
        print("❌ demo.py not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if main dependencies are available
    try:
        import pandas
        import matplotlib
        print("✅ Core dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("🚀 Launching interactive demo...")
    print()
    
    # Launch the main demo
    try:
        subprocess.run([sys.executable, 'demo.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed to launch: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")

if __name__ == "__main__":
    main()