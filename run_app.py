import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print(f"Iniciando aplicación: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
