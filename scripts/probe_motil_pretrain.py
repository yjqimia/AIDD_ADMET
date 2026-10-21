"""Small diagnostic of the unmodified upstream pretraining call, not full pretraining."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.upstream = a.upstream.resolve()
a.out = a.out.resolve()
os.chdir(a.upstream / 'MotiL_micromolecule')
sys.path.insert(0, str(a.upstream / 'MotiL_micromolecule'))
from chemprop.parsing import add_train_args, modify_train_args
from chemprop.models import build_pretrain_model
from chemprop.train.run_training import DDPM, MLP

parser = argparse.ArgumentParser()
add_train_args(parser)
args = parser.parse_args(['--exp_name', 'diagnostic', '--exp_id', 'single_ethanol',
                          '--data_path', './data/zinc15_250K.csv', '--step', 'pretrain',
                          '--dataset_type', 'classification', '--no_cuda'])
modify_train_args(args)
model = build_pretrain_model(args, encoder_name='CMPNN')
diffusion = DDPM(model, MLP(300, 300, 50), 1000)
result = {'scope': 'One ethanol molecule through the upstream DDPM loss, CPU diagnostic only',
          'full_pretraining_executed': False}
try:
    loss = diffusion.compute_loss(['CCO'], 'pretrain', 'contrast_mol', args.depth, args.dropout1)
    result.update(status='completed', loss=float(loss.detach()))
except Exception as e:
    result.update(status='failed', exception_type=type(e).__name__, error=str(e), traceback=traceback.format_exc())
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
# A failed upstream operation stays a failed shell command, even though its report is saved.
sys.exit(0 if result['status'] == 'completed' else 1)
