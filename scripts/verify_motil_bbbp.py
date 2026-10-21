"""Re-evaluate each saved best checkpoint and audit actual split membership."""
import argparse
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--device', default='cpu', help='Inference device: cpu or cuda:N')
a = p.parse_args()
a.upstream = a.upstream.resolve()
a.run_dir = a.run_dir.resolve()
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=True)
os.chdir(a.upstream / 'MotiL_micromolecule')
sys.path.insert(0, str(a.upstream / 'MotiL_micromolecule'))
from chemprop.utils import load_checkpoint
from chemprop.data.utils import get_data
from chemprop.train.predict import predict

torch.set_num_threads(2)
device = torch.device(a.device)
if device.type not in ('cpu', 'cuda'):
    raise ValueError('Choose cpu or cuda:N.')
if device.type == 'cuda':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback.')
    torch.cuda.set_device(device)
RDLogger.DisableLog('rdApp.*')
log = (a.run_dir / 'train.log').read_text()
logged_scores = [float(v) for v in re.findall(r'Ensemble test auc = ([0-9.]+)', log)]
if len(logged_scores) != 3 or '3-time runs' not in log or 'Overall test auc' not in log:
    raise RuntimeError('Three completed runs are required; do not validate partial outputs.')
initial_path = a.upstream / 'MotiL_micromolecule/dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl'
initial = torch.load(initial_path, map_location='cpu', weights_only=True)
records = []
curves, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for i in range(3):
    run = a.run_dir / f'run_{i}'
    ckpt = run / 'model_0/model.pt'
    saved = torch.load(ckpt, map_location='cpu')
    model = load_checkpoint(str(ckpt), cuda=device.type == 'cuda', logger=logging.getLogger('verification'))
    data = get_data(path=str(run / 'test_full.csv'), args=saved['args'])
    preds = np.array(predict(model, data, batch_size=16))[:, 0]
    targets = np.array(data.targets())[:, 0]
    assert len(preds) > 0 and np.isfinite(preds).all()
    assert set(targets) == {0, 1}
    auc = float(roc_auc_score(targets, preds))
    assert abs(auc-logged_scores[i]) < 1e-6, (i, auc, logged_scores[i])
    with (a.out / f'run_{i}_test_predictions.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['smiles', 'label', 'predicted_probability'])
        w.writerows(zip(data.smiles(), targets.tolist(), preds.tolist()))
    splits, scaffold_sets, sizes = {}, {}, {}
    for name in ['train', 'val', 'test']:
        path = run / f'{name}_full.csv'
        splits[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline='') as f:
            rows = list(csv.reader(f))[1:]
        sizes[name] = len(rows)
        scaffold_sets[name] = {MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(r[0]),
                                                               includeChirality=False) for r in rows}
    overlap = {f'{x}_{y}': len(scaffold_sets[x] & scaffold_sets[y])
               for x, y in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    assert not any(overlap.values()), overlap
    encoder_keys = model.encoder.state_dict()
    compatible = [k for k, v in initial.items() if k in encoder_keys and v.shape == encoder_keys[k].shape]
    assert len(compatible) == len(encoder_keys) == len(initial)
    segment = log.split(f' - Run {i}\n', 1)[1]
    segment = segment.split(f' - Run {i+1}\n', 1)[0]
    epochs = re.findall(r'Epoch: (\d+), auc \(train:val:test\) = ([0-9.]+), ([0-9.]+), ([0-9.]+)', segment)
    assert len(epochs) == 10, (i, len(epochs))
    values = np.array(epochs, dtype=float)
    best = re.search(r'best validation auc = ([0-9.]+) on epoch (\d+)', segment)
    assert best is not None
    for c, name in [(1, 'train'), (2, 'validation'), (3, 'test (monitor only)')]:
        axes[i].plot(values[:, 0], values[:, c], label=name)
    axes[i].axvline(int(best[2]), color='black', linestyle='--', alpha=0.5, label='selected by validation')
    axes[i].set(title=f'Run {i}; split seed {saved["args"].seed}', xlabel='Epoch (zero-based)', ylim=(0.8, 1))
    records.append({'run': i, 'actual_split_seed': saved['args'].seed,
                    'sizes': sizes, 'test_roc_auc': auc,
                    'best_validation_auc_logged': float(best[1]), 'best_epoch_zero_based': int(best[2]),
                    'split_sha256': splits, 'cross_split_scaffold_overlap': overlap,
                    'initial_checkpoint_compatible_encoder_tensors': len(compatible),
                    'checkpoint_sha256': hashlib.sha256(ckpt.read_bytes()).hexdigest()})
scores = [r['test_roc_auc'] for r in records]
axes[0].set_ylabel('ROC-AUC')
axes[2].legend(fontsize=7, loc='lower right')
training_device = 'CUDA' if saved['args'].cuda else 'CPU'
curves.suptitle(f'{training_device} fine-tuning runs using the author checkpoint; not full pretraining')
curves.tight_layout()
curves.savefig(a.out / 'learning_curves.png', dpi=170)
plt.close(curves)
summary = {'upstream_commit': subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip(),
           'training_device': training_device, 'verification_device': str(device),
           'torch': torch.__version__, 'epochs_per_run': 10, 'completed_runs': 3,
           'mean_roc_auc': float(np.mean(scores)), 'std_roc_auc_ddof0': float(np.std(scores)),
           'all_runs_identical_splits': all(r['split_sha256'] == records[0]['split_sha256'] for r in records),
           'runs': records,
           'limitations': ['Compatibility environment; not the exact paper environment.',
                          'The upstream launcher uses the same split seed for all runs despite incremented seed labels.',
                          'Uses an author-supplied initial checkpoint; full pretraining is not reproduced.']}
(a.out / 'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
