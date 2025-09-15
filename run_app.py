#!/usr/bin/env python3
"""Run the Streamlit application for plant disease detection."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit app."""
    app_path = Path(__file__).parent / "src" / "streamlit_app" / "app.py"
    
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)
    
    print("🌱 Starting Plant Disease Detection App...")
    print("📱 Open your browser to view the application")
    print("🔗 The app will be available at: http://localhost:8501")
    print("\n" + "="*50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
