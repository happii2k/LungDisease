import subprocess
import sys

def run():
    if sys.platform == "win32":
        # Windows
        cmd = 'source .venv/bin/activate && pip install -r requirements.txt && python train.py'
        subprocess.call(cmd, shell=True)
    else:
        # Linux/Mac
        cmd = 'source .venv/bin/activate && pip install -r requirements.txt && python train.py'
        subprocess.call(cmd, shell=True, executable='/bin/bash')

run()
