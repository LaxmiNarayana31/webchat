from pathlib import Path
import subprocess
import sys
from typing import Optional


def launch_streamlit(port: Optional[int] = None) -> None:
    """Launches the Streamlit analytical dashboard in a dedicated subprocess."""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    streamlit_file = str(root_dir / "streamlit_app" / "app.py")
    port_arg = ["--server.port", str(port)] if port else []
    cmd = [sys.executable, "-m", "streamlit", "run", streamlit_file] + port_arg
    print(f"[WebChat] Starting Streamlit Dashboard ({' '.join(cmd)})...")
    sys.exit(subprocess.call(cmd))
