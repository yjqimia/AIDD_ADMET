# AIDD_ADMET

分子性质预测与 ADMET 论文阅读、复现记录。

## MotiL

- [文章阅读：只含 Abstract 和 Results](notes/文章阅读.md)
- [复现记录：环境、每步命令、结果与未完成事项](notes/复现记录.md)
- [本机 RTX 3080：安装、运行、记录过程并同步 GitHub](notes/本机复现与GitHub同步.md)
- [材料清单与 SHA256](notes/materials_manifest.json)
- [材料核查、环境与作者 Source Data 再分析](results/motil_audit)

论文：Liu et al., *Molecular Motif Learning as a pretraining objective for molecular property prediction*, Nature Communications (2025), [DOI](https://doi.org/10.1038/s41467-025-66685-w)。

复现分阶段进行；作者结果再分析、自己的模型运行与完整论文复现分别记录。大数据、PDF、模型和隔离环境保存在库外，未将上游代码拷入本库。

## 本地同步

云端工作区与个人电脑相互独立。更新推送到 GitHub 后，在本地本库的 `main` 分支、工作区干净时执行：

```bash
git pull --ff-only origin main
```

同步会获取笔记、脚本和已入库结果；本地的 Python 环境、作者源码、数据和模型仍需按复现记录单独准备，云端安装不会自动出现在个人电脑上。
