# 在本机 RTX 3080 上复现，并同步代码、过程和结果

这份说明面向你的 Linux 本机 Terminal，沿用之前记录的 `/home/yjqi/projects1` 项目目录；实际位置不同时替换目录。云端与本机是两台机器，GitHub 负责同步文件，不会把云端 Python 环境或计算任务迁移到你的电脑。

**当前第一阶段目标**：用作者的预训练权重，在本机 GPU 上完成 BBBP 的 10 epochs × 3 runs 微调、独立模型验证和完整记录。不是从头预训练，也不代表整篇论文已复现。作者预训练接口问题、三轮相同 split seed 的问题仍保留，见[复现记录](复现记录.md)。

## 1. 三类文件分别放在哪里？

| 内容 | 本机位置（相对本库根目录） | 是否同步普通 Git |
|---|---|---|
| 自己的代码、环境清单、操作说明 | `scripts/`、`notes/` | 是 |
| 每次实验的过程与轻量结果 | `results/local/<run-id>/` | 是 |
| 作者固定版本源码、数据、完整模型、下载缓存 | `.local/motil/` | 默认忽略，保留本机 |

每个实验目录包含：`README.md`（逐条命令及退出码）、`commands.json`、`manifest.json`、实际硬件、环境版本、训练与验证日志、三轮预测 CSV、划分 CSV、学习曲线、指标和权重 SHA256。

完整 checkpoint 的路径会登记，文件留在 `.local/motil/runs/`，不会因推送笔记而丢失。**同步轻量结果不等于把完整模型也上传了**；如需跨电脑共享 checkpoint，可另用 GitHub Release 或 Git LFS，并保持文件哈希一致。

## 2. 先把最新代码拉到本机

已 clone 本库时，在你的 Terminal 执行：

```bash
cd /home/yjqi/projects1/AIDD_ADMET
git status --short
git switch main
git pull --ff-only origin main
```

如果 `git status` 有自己的未提交改动，先保存、提交或暂存这些改动再切换／拉取，不使用 reset/clean 删除它们。

如果还没有 clone 本库，改为执行下面这一组；不要对已有同名目录重复 clone：

```bash
cd /home/yjqi/projects1
git clone https://github.com/yjqimia/AIDD_ADMET.git
cd AIDD_ADMET
```

## 3. 准备独立 GPU 环境（首次安装）

这些命令在**本机 Terminal**执行，不是在聊天窗口或云端执行：

```bash
nvidia-smi
conda create -n motil-repro python=3.9.25 pip -y
conda activate motil-repro
bash scripts/setup_motil_gpu.sh
```

若已创建 `motil-repro`，不再重复创建，直接 `conda activate motil-repro`。每次新开 Terminal 都要先激活，再进入本库目录。

安装目标为：Python 3.9.25、PyTorch 2.2.2 + CUDA 12.1、RDKit 2022.9.1、PyG 2.3.0、scikit-learn 0.24.2、torch-scatter 2.1.2。其他依赖完整固定于 [requirements-motil-local.txt](../scripts/requirements-motil-local.txt)。这是兼容环境，不是论文原始 PyTorch 1.12.1 环境。

你之前记录的 535 系列驱动支持这条 CUDA 12.1 运行路线；仍以本次 `nvidia-smi` 实测为准。官方 PyTorch 和 scatter 二进制包提供运行组件，这条路线不要求重新编译 CUDA 扩展或更改系统 CUDA Toolkit。不要运行云端的 `setup_motil_cpu.sh`。

成功输出必须包括：

- 检测到的显卡名称，你的机器上应是 RTX 3080。
- `torch.cuda.is_available()` 为真。
- 在 GPU 上执行 512×512 矩阵运算且结果有限。
- 作者代码固定在 `2a8f006d1b72d17e5bc6759e72f19b21f2b5aa64`，`train.py --help` 可以运行。

安装日志位于 `.local/motil/setup/<UTC时间>-<进程号>/setup.log`；成功后运行器会把最近一次安装日志复制进实验结果目录。若安装中途失败，保留该日志并根据具体错误处理，不要把安装脚本输出了一部分当成环境成功。

如果安装失败，还没进入训练，也能把过程留档。将下面的路径替换为终端打印的实际日志路径，并写清失败原因：

```bash
mkdir -p results/local/setup_failure_001
cp .local/motil/setup/实际UTC时间-进程号/setup.log results/local/setup_failure_001/
```

在该目录写 `README.md`，记录安装命令、报错、修改和下一步，然后按第 5 节提交这个目录。不要把失败的安装记为已经在 GPU 上训练。

所需下载站点包括 GitHub、PyPI、`download.pytorch.org`、`data.pyg.org`。本阶段的数据与权重来自作者 GitHub checkout，不依赖 Hugging Face。若访问失败，请保留具体报错，不关闭 TLS 验证。

## 4. 用自己的 3080 正式运行

确认自己位于本库根目录、已激活环境，运行：

```bash
conda activate motil-repro
python scripts/run_motil_local.py --run-id bbbp_3080_001 --gpu 0
```

`--gpu 0` 是进程可见的第一张 CUDA 显卡，程序会记录实际名称。**不允许悄悄回退到 CPU**。运行器调用作者原始 `train.py`，使用 `--gpu 0`，没有 `--no_cuda`。

运行器自动完成：

1. 检查 GPU 和有限值运算；记录硬件、代码 commit、原始 CSV 和权重哈希。
2. 检查自己的已跟踪代码与作者 checkout 没有未提交差异，避免“代码版本号与实际执行代码不一致”。如果你修改了代码，先提交代码，再开新的实验。
3. 执行作者 BBBP 配置的三轮微调，保留真实训练退出码。
4. 复制参数、训练日志和三轮划分；在同一 GPU 上重新加载选中的模型，计算测试预测与 ROC-AUC，核查 scaffold 是否跨组。
5. 生成 `results/local/bbbp_3080_001/README.md`，写出每条实际调用命令、工作目录、退出码、成功结果或失败原因。

查看结果：

```bash
cat results/local/bbbp_3080_001/README.md
cat results/local/bbbp_3080_001/manifest.json
cat results/local/bbbp_3080_001/verification.json
```

只有 `manifest.json` 的 `status` 为 `completed`，且验证日志退出 0，才记为本机运行成功。失败或中断也会保留过程笔记；可能没有 `verification.json`，不能照抄云端文件补上。

已有同名实验目录时，程序会拒绝覆盖。需要重试时选择新 ID：

```bash
python scripts/run_motil_local.py --run-id bbbp_3080_002 --gpu 0
```

这条是重试示例，不要求连续重复做两套实验。不要在训练进程运行期间编辑它正在使用的代码或启动脚本。

目前仍保留作者三轮 **split seed 都为 2024** 的行为。程序会如实记录，不能将这三次运行称为三个不同的数据划分。将来改为三个不同划分时，需要另开实验编号，记录新的代码与参数，不能覆盖这次结果。

## 5. 把代码、过程和结果提交到 GitHub

先看实验 README 与日志，再在本机仓库执行：

```bash
git status --short
git add results/local/bbbp_3080_001
git diff --cached --stat
git commit -m "Record local RTX 3080 BBBP experiment 001"
git push origin main
```

这会把该实验的 README、命令、环境、日志、CSV、图表和指标一起同步。失败实验也可以用注明失败的 commit 留档；不要只挑成功分数、不保留失败过程。

若你自己改了代码或笔记，**应在运行前**先提交相关文件，使实验记录中的 commit 指向实际代码。例如：

```bash
git add scripts/run_motil_local.py scripts/requirements-motil-local.txt
git diff --cached
git commit -m "Adjust local MotiL experiment configuration"
```

只添加你实际修改且打算保存的文件。新代码也必须先加入并提交；未跟踪文件不会因为 `git diff` 显示干净就自动成为实验代码版本的一部分。自己的实验代码优先放在本库 `scripts/`；`.local` 中作者 checkout 是独立仓库，不会通过主库的一次 `git add` 自动同步。

如果 push 提示远端已有更新，先确认本地工作区干净，再执行：

```bash
git pull --rebase origin main
git push origin main
```

有冲突时先解决，不强制推送。GitHub 本地登录由你机器已有的 HTTPS／SSH 凭据处理，遇到身份错误时保留错误信息，不把 token 写进笔记或脚本。

## 6. 以后每天怎样工作？

开始：进入本库 → 保存自己的改动 → 拉取最新代码 → 激活 `motil-repro`。

实验：先提交使用的代码 → 使用新 run ID 运行 → 检查自动生成的实验笔记与结果。

结束：提交这次结果和补充解释 → `git push origin main`。以后我可以根据你推送的日志继续分析，你也能在另一台电脑通过 pull 看到相同笔记；但另一台电脑仍需单独准备计算环境。

## 7. 当前验证状态

此流程由云端编写。**我无法从云端操作你本机的 RTX 3080，尚未在你的本机安装或执行 GPU 实验。** 你执行第 3、4 节后生成的硬件和实验日志，才是本机验证证据。

已在当前云端检查脚本语法、验证无 GPU 时会失败并保留日志而不切换到 CPU，并用已有三个 CPU checkpoint 回归验证通用结果检查器。GPU 安装、CUDA wheel 下载和 GPU 训练仍需在本机实测；不把云端 CPU 检查冒充 GPU 测试。

## 8. GPU 安装脚本的完整命令

下面收录 [setup_motil_gpu.sh](../scripts/setup_motil_gpu.sh) 的全文；每次实验实际调用的训练与验证命令由运行器写入该次实验的 README 与 commands.json。

```bash
#!/usr/bin/env bash
# Run in the local motil-repro conda environment, not the cloud CPU environment.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
study_dir="$repo_dir/.local/motil"
export PYTHONDONTWRITEBYTECODE=1
if [[ -z ${CONDA_PREFIX:-} && -z ${VIRTUAL_ENV:-} ]]; then
    echo 'Activate a dedicated Python 3.9 conda/venv environment first.' >&2
    exit 1
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 9), "Use Python 3.9 (recommended 3.9.25)"'
command -v nvidia-smi > /dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
setup_dir="$study_dir/setup/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$setup_dir"
exec > >(tee "$setup_dir/setup.log") 2>&1
printf 'Setup log: %s/setup.log\n' "$setup_dir"
run() {
    printf '\n$ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}
run python --version
run nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
run python -m pip install 'torch==2.2.2+cu121' --index-url https://download.pytorch.org/whl/cu121
run python -m pip install -r "$repo_dir/scripts/requirements-motil-local.txt"
# Official binary wheels match torch 2.2.x / CUDA 12.1; no local nvcc build required.
run python -m pip install --no-index --only-binary=:all: 'torch-scatter==2.1.2+pt22cu121' \
    --find-links https://data.pyg.org/whl/torch-2.2.0+cu121.html
run python -c "import torch, torch_scatter, torch_geometric, rdkit, sklearn, tkinter; assert torch.cuda.is_available(), 'CUDA unavailable; do not start the GPU experiment'; x=torch.randn(512,512,device='cuda:0'); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all().item(); print('GPU:',torch.cuda.get_device_name(0)); print('torch:',torch.__version__,'CUDA runtime:',torch.version.cuda,'scatter:',torch_scatter.__version__,'PyG:',torch_geometric.__version__,'RDKit:',rdkit.__version__,'sklearn:',sklearn.__version__)"
run python -c "import torch; from torch_scatter import scatter_add; x=torch.ones(4,device='cuda:0'); i=torch.tensor([0,0,1,1],device='cuda:0'); y=scatter_add(x,i,dim=0); assert torch.equal(y,torch.tensor([2.,2.],device='cuda:0')); print('CUDA scatter kernel OK')"
upstream_commit=2a8f006d1b72d17e5bc6759e72f19b21f2b5aa64
if [[ ! -e "$study_dir/MotiL" ]]; then
    run git clone https://github.com/Young0222/MotiL.git "$study_dir/MotiL"
    run git -C "$study_dir/MotiL" checkout --detach "$upstream_commit"
fi
if [[ $(git -C "$study_dir/MotiL" rev-parse HEAD) != "$upstream_commit" ]]; then
    echo 'Existing author checkout has another HEAD; preserve it and inspect before proceeding.' >&2
    exit 1
fi
run git -C "$study_dir/MotiL" diff --exit-code HEAD
cd "$study_dir/MotiL/MotiL_micromolecule"
run python train.py --help
python -m pip freeze > "$setup_dir/environment.freeze.txt"
printf '%s\n' "$setup_dir" > "$study_dir/last_setup_path.txt"
printf '\nSetup complete. Log: %s/setup.log\n' "$setup_dir"
```
