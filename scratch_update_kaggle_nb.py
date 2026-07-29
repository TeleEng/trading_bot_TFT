import json

nb_path = 'd:/Work/trading_bot/kaggle_pull/trading-bot-tft-pipeline-2026.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": [
    "# Copy data from Kaggle dataset to bypass downloading\n",
    "!mkdir -p /kaggle/working/trading_bot_TFT/data\n",
    "!cp -r /kaggle/input/*/trading_bot_TFT/data/* /kaggle/working/trading_bot_TFT/data/ || echo 'Data copy from dataset failed'\n",
    "!ls /kaggle/working/trading_bot_TFT/data"
  ]
}

# Find the cell that does os.chdir('trading_bot_TFT') and insert after it
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "os.chdir('trading_bot_TFT')" in source or "os.chdir(\"trading_bot_TFT\")" in source:
            nb['cells'].insert(i + 1, new_cell)
            break

# Update the first cell's comment to ensure uniqueness for Kaggle push
if len(nb['cells']) > 1 and nb['cells'][1]['cell_type'] == 'code':
    nb['cells'][1]['source'].insert(0, '# Added dataset copy step\n')

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
