"""Run and document a local CUDA BBBP experiment without editing author code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
UPSTREAM_SHA = '2a8f006d1b72d17e5bc6759e72f19b21f2b5aa64'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-id', required=True, help='New identifier, e.g. bbbp_3080_001')
p.add_argument('--gpu', type=int, default=0, help='Logical CUDA device number')
p.add_argument('--study-dir', type=Path, default=REPO / '.local/motil')
p.add_argument('--results-dir', type=Path, default=REPO / 'results/local')
a = p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', a.run_id) or a.gpu < 0:
    p.error('Use an alphanumeric run ID (with _ or -), and a nonnegative GPU number.')
a.study_dir = a.study_dir.resolve()
out = a.results_dir.resolve() / a.run_id
out.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing experiment.
upstream = a.study_dir / 'MotiL'
micro = upstream / 'MotiL_micromolecule'
commands = []
manifest = {'run_id': a.run_id, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'running', 'requested_device': f'cuda:{a.gpu}', 'full_pretraining': False}
environment = dict(os.environ)
environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', MPLCONFIGDIR=str(a.study_dir / 'mpl'))


def save():
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    (out / 'commands.json').write_text(json.dumps(commands, ensure_ascii=False, indent=2)+'\n')
    write_note()


def run(args, cwd, filename):
    args = [str(v) for v in args]
    item = {'argv': args, 'cwd': str(cwd), 'log': filename,
            'started_utc': datetime.now(timezone.utc).isoformat()}
    commands.append(item)
    save()
    print(f'\n[{cwd}] $ {shlex.join(args)}', flush=True)
    output = []
    with (out / filename).open('w') as log:
        try:
            process = subprocess.Popen(args, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as e:
            item['launch_error'] = str(e)
            log.write(str(e)+'\n')
            save()
            raise
        with process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    output.append(line)
                code = process.wait()
            except KeyboardInterrupt:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                item['exit_code'] = 130
                save()
                raise
    item['exit_code'] = code
    item['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    if code:
        raise subprocess.CalledProcessError(code, args)
    return ''.join(output)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write_note():
    text = [f'# 本机实验：{a.run_id}', '',
            f"状态：**{manifest['status']}**；开始时间（UTC）：{manifest['started_utc']}。", '',
            '这是请求在本机 CUDA 上执行的任务；具体硬件以 hardware.json 为准。'
            '没有硬件验证文件或状态不为 completed 时，不计为 GPU 复现成功。', '',
            f"代码 commit：`{manifest.get('study_code_commit', '未获取')}`；"
            f"作者 commit：`{manifest.get('upstream_commit', '未获取')}`。", '',
            '使用作者权重，10 epochs × 3 runs，保留作者启动器的相同 split seed=2024 行为。'
            '这不是从头预训练，也不是三个不同的 scaffold 划分。', '',
            '## 实际执行的每条命令', '']
    for item in commands:
        status = '未启动（系统错误）' if 'launch_error' in item else item.get('exit_code', '尚未结束')
        text += [f"退出码：`{status}`；日志：[{item['log']}]({item['log']})", '',
                 '```bash', f"cd {shlex.quote(item['cwd'])}", shlex.join(item['argv']), '```', '']
    if 'error' in manifest:
        text += ['## 问题', '', '```text', manifest['error'], '```', '',
                 '保留日志与本 ID 的文件，解决问题后使用新的 run ID；不删除旧结果伪装成首次成功。', '']
    if (out / 'verification.json').exists():
        result = json.loads((out / 'verification.json').read_text())
        text += ['## 独立验证', '',
                 f"ROC-AUC：**{result['mean_roc_auc']:.9f} ± {result['std_roc_auc_ddof0']:.9f}**（ddof=0）。", '',
                 f"三轮实际划分相同：`{result['all_runs_identical_splits']}`。", '',
                 '[指标、划分和权重哈希](verification.json)；[环境](environment.freeze.txt)；'
                 '[硬件](hardware.json)；[完整步骤与退出码](commands.json)。', '',
                 '![学习曲线](learning_curves.png)', '']
    text += ['## 文件与同步', '',
             '本目录保存日志、参数、CSV 预测、划分、图片、环境版本和代码版本，可提交到 GitHub。'
             '完整 checkpoint 保留在 .local/motil/runs 下（或自定义 study-dir），不自动放入普通 Git。', '',
             'Git 提交和推送由使用者在检查结果后执行；本程序不会自动推送或读取凭据。', '']
    (out / 'README.md').write_text('\n'.join(text))


code = 1
try:
    save()
    probe = (
        'import json,sys,platform; from pathlib import Path; import torch; '
        "assert torch.cuda.is_available(), 'CUDA unavailable: no CPU fallback'; "
        "assert torch.__version__.split('+')[0] == '2.2.2' and torch.version.cuda == '12.1', 'Activate the prepared torch 2.2.2 / CUDA 12.1 environment'; "
        'gpu=int(sys.argv[1]); torch.cuda.set_device(gpu); '
        'x=torch.randn(512,512,device=f"cuda:{gpu}"); y=x@x; torch.cuda.synchronize(); '
        "assert torch.isfinite(y).all().item(), 'GPU matrix check failed'; "
        'd=dict(platform=platform.platform(),python=sys.version,torch=torch.__version__,'
        'cuda_runtime=torch.version.cuda,gpu_index=gpu,gpu_name=torch.cuda.get_device_name(gpu),'
        'gpu_total_bytes=torch.cuda.get_device_properties(gpu).total_memory); '
        'Path(sys.argv[2]).write_text(json.dumps(d,indent=2)); print(json.dumps(d,indent=2))'
    )
    run([sys.executable, '-c', probe, a.gpu, out / 'hardware.json'], REPO, 'gpu_check.log')
    manifest['study_code_commit'] = run(['git', 'rev-parse', 'HEAD'], REPO, 'study_commit.log').strip()
    run(['git', 'diff', '--exit-code', 'HEAD'], REPO, 'study_diff.log')
    manifest['upstream_commit'] = run(['git', 'rev-parse', 'HEAD'], upstream, 'upstream_commit.log').strip()
    if manifest['upstream_commit'] != UPSTREAM_SHA:
        raise RuntimeError('Author HEAD differs from the fixed experiment. Inspect before running.')
    run(['git', 'diff', '--exit-code', 'HEAD'], upstream, 'upstream_diff.log')
    run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], REPO, 'driver.txt')
    (out / 'environment.freeze.txt').write_text(run([sys.executable, '-m', 'pip', 'freeze'], REPO, 'pip_freeze.log'))
    setup_path = a.study_dir / 'last_setup_path.txt'
    if setup_path.exists():
        setup_log = Path(setup_path.read_text().strip()) / 'setup.log'
        if setup_log.is_file():
            shutil.copy2(setup_log, out / 'setup.log')
    checkpoint = micro / 'dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl'
    manifest['inputs_sha256'] = {'bbbp.csv': sha(micro / 'data/bbbp.csv'),
                               'initial_checkpoint': sha(checkpoint)}
    dump = a.study_dir / 'runs' / a.run_id
    if dump.exists():
        raise FileExistsError(f'Checkpoint directory already exists: {dump}')
    train = [sys.executable, 'train.py', '--data_path', './data/bbbp.csv',
             '--exp_id', a.run_id, '--dataset_type', 'classification', '--epochs', '10',
             '--num_runs', '3', '--gpu', a.gpu, '--batch_size', '16', '--seed', '2024',
             '--init_lr', '1e-5', '--final_lr', '5e-6', '--wd', '5e-5', '--warmup_epochs', '0.0',
             '--split_type', 'scaffold_balanced', '--exp_name', 'finetune',
             '--split_sizes', '0.8', '0.1', '0.1', '--step', 'finetune', '--save_smiles_splits',
             '--dump_path', dump, '--checkpoint_path', checkpoint]
    run(train, micro, 'training_console.log')
    candidates = list(dump.glob(f'*-finetune/{a.run_id}'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected exactly one author run directory, found {len(candidates)}.')
    run_dir = candidates[0]
    manifest['checkpoint_directory'] = str(run_dir)
    shutil.copy2(run_dir / 'train.log', out / 'train.log')
    shutil.copy2(run_dir / 'params.pkl', out / 'params.json')
    for i in range(3):
        target = out / 'splits' / f'run_{i}'
        target.mkdir(parents=True)
        for name in ['train_full.csv', 'val_full.csv', 'test_full.csv']:
            shutil.copy2(run_dir / f'run_{i}' / name, target / name)
    run([sys.executable, REPO / 'scripts/verify_motil_bbbp.py', '--upstream', upstream,
         '--run-dir', run_dir, '--out', out, '--device', f'cuda:{a.gpu}'], REPO, 'verification.log')
    manifest['status'] = 'completed'
    code = 0
except KeyboardInterrupt:
    manifest.update(status='interrupted', error='Interrupted by user; partial outputs retained.')
    code = 130
except Exception as e:
    manifest.update(status='failed', error=f'{type(e).__name__}: {e}')
    (out / 'failure.txt').write_text(traceback.format_exc())
    print(manifest['error'], file=sys.stderr)
    if isinstance(e, subprocess.CalledProcessError) and e.returncode > 0:
        code = e.returncode
finally:
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"\nStatus: {manifest['status']}; notes and logs: {out / 'README.md'}", flush=True)
sys.exit(code)
