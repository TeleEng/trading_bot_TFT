import json
import os
import sys
import subprocess

with open(r'C:\Users\Sadegh\.kaggle\kaggle.json', 'r') as f:
    creds = json.load(f)

env = os.environ.copy()
env['KAGGLE_USERNAME'] = creds['username']
env['KAGGLE_KEY'] = creds['key']

cmd = [r'C:\Users\Sadegh\AppData\Local\anaconda3\envs\sumo-rl\Scripts\kaggle.exe', 'kernels', 'push', '-p', 'd:/Work/trading_bot/notebooks']
result = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
