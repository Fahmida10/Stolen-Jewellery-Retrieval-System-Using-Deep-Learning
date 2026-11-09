# GitLab Upload Notes

This directory is the clean GitLab upload package for the jewelry retrieval project.

## Included

- Model code in the `01_resnet18` through `06_clip_l14` folders
- Model-specific README files
- Model-specific architecture and result figures inside each stage folder's `figures/` directory

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
