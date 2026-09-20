"""
Root Entry Point for Streamlit Cloud & Standalone Deployment.
Routes execution directly to streamlit_app/app.py.
"""
from pathlib import Path
import runpy
import sys

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute the main Streamlit application
APP_PATH = ROOT_DIR / "streamlit_app" / "app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")

