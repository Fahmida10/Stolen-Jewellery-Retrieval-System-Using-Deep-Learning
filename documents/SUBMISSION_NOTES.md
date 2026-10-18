# GitLab Upload Notes

**Navigate:** [Portfolio home](../README.md) · [Site map](../README.md#site-map) · [Documents index](README.md) · [Technical report](README_DETAILED_REPORT.md#table-of-contents)

This directory is the clean GitLab upload package for the jewelry retrieval project.

## Included

- Model code in the `code/01_resnet18` through `code/06_clip_l14` folders
- Model-specific README files
- Model-specific architecture and result figures consolidated under the portfolio folder `images/` (filenames prefixed by stage, for example `stage06_clip_l14_*.png`)

## Excluded

- Pickle files: `*.pkl`, `*.pickle`
- Model checkpoints and weights: `checkpoints/`, `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`
- Embedding dumps: `embeddings/`, `*.npy`, `*.npz`
- Local environments and caches: `venv/`, `.venv/`, `__pycache__/`, `*.pyc`
- TensorBoard/runtime logs and temporary LaTeX build files

Before uploading, you can re-run the artifact scan from this directory:

```bash
find . -type f \( -iname '*.pkl' -o -iname '*.pickle' -o -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' -o -iname '*checkpoint*' \) -print
```

The command should print nothing.
