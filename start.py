# This script should be run from the root of your workspace

import subprocess
import sys
import os

def run(cmd, check=True, shell=False):
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    subprocess.run(cmd, check=check, shell=shell)

# 1. Create uv virtual environment
run(["uv", "venv", ".venv"])

# 2. Install dependencies in the venv
run(["uv", "pip", "install", "-e", "."])

# 3. Activate the virtual environment and run Streamlit in the same CLI
if os.name == "nt":
    # Use venv's python executable to run Streamlit
    streamlit_exe = os.path.join(".venv", "Scripts", "python.exe")
    run([streamlit_exe, "-m", "streamlit", "run", "main.py"])
else:
    activate_script = ".venv/bin/activate"
    run(f"source {activate_script} && streamlit run main.py", shell=True)

print("LLM UI started.")
if os.name != "nt":
    print("To attach: screen -r llmui")