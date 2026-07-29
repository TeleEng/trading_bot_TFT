import json

nb_path = 'd:/Work/trading_bot/notebooks/kaggle_full_pipeline.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['source'].insert(0, '# Force run with T4 GPU instead of P100\n')
        break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
