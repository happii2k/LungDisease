
import os
def run():
    command = ( [
                "python -m venv .venv",

                "source .venv/bin/activate",

                "pip install -r requirements.txt",

                "python train.py"
                ])
    for com in command:
      os.system(com)

run()