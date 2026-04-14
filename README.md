# STAD Training Pipeline

This repository now contains a PyTorch implementation scaffold for **STAD**:

- a DreamDiffusion-style EEG MAE backbone for HR EEG latents,
- an STC conditioner for LR EEG plus electrode geometry,
- an MTD latent denoiser trained with a DDPM-style noise-prediction objective,
- subject-wise Localize-MI loaders built around the bundled `.pt` files in `dataset/localizeMI_unprocessed/`.

## Important Data Reality Check

The bundled data already differs from the paper preprocessing:

- `7` subject files are present locally
- epochs are already saved as tensors of shape `[256, 1280]`
- metadata reports `256 Hz`, `resampled=True`, `z_score_normalized=True`
- metadata does **not** record the paper's reported 50/100/150/200 Hz notch chain

The code treats these as explicit dataset deviations and writes them into the generated dataset report.

## Project Layout

```text
configs/
data/
models/
training/
eval/
utils/
trainer.py
outline.md
```

## Main Commands

The files under `configs/` keep the `.yaml` suffix for readability, but they currently use JSON-compatible syntax so the pipeline can run in a minimal Python environment without extra config-parser dependencies.

Materialize a dataset report from the bundled subject files:

```bash
python3 trainer.py preprocess --config configs/stad_localize_mi_scale4.yaml
```

Pretrain the EEG MAE:

```bash
python3 trainer.py mae --config configs/stad_localize_mi_scale4.yaml --device cuda
```

Train STAD:

```bash
python3 trainer.py stad --config configs/stad_localize_mi_scale4.yaml --device cuda
```

Evaluate reconstruction:

```bash
python3 trainer.py eval --config configs/stad_localize_mi_scale4.yaml --device cuda
```

## DreamDiffusion MAE Weights

The config supports two MAE initialization paths:

1. Local MAE checkpoint at `model.mae.pretrained_checkpoint_path`
2. Download URL at `model.mae.pretrained_checkpoint_url`

If the file path is missing and a URL is provided, the code will download the checkpoint before loading it.

The default configs point to:

```text
checkpoints/dreamdiffusion_mae_checkpoint.pth
```

Set the URL once you have the official DreamDiffusion checkpoint location you want to use.

## Notes On Faithfulness

This implementation follows the STAD paper at the module level, but a few choices remain assumptions because the paper and bundled data leave gaps:

- LR montages are generated deterministically from 256-channel coordinates with nested farthest-point subsampling.
- The bundled data is already preprocessed, so the raw filtering/epoching stage cannot be reproduced from recordings here.
- STC aggregates channel-wise temporal features into time-aligned condition tokens instead of keeping a full channel-time token grid.
- The MAE interface is intentionally ViT-MAE-like so official DreamDiffusion checkpoints have a workable loading path.

## Smoke Test

```bash
pytest -q
```

## Docker

Build and run everything from a sourceable shell helper:

```bash
source env.sh
```

By default, `source env.sh` now launches an interactive shell inside the Docker container.

If you only want to load the helper functions into your host shell without entering Docker immediately:

```bash
source env.sh --no-shell
stad_doctor
stad_build
stad_shell
```

Useful commands after `source env.sh`:

```bash
stad_preprocess --config configs/stad_localize_mi_scale4.yaml
stad_train_mae --config configs/stad_localize_mi_scale4.yaml --device cpu
stad_train_stad --config configs/stad_localize_mi_scale4.yaml --device cpu
stad_eval --config configs/stad_localize_mi_scale4.yaml --device cpu
stad_test
```

Notes:

- The image installs both `requirements.txt` and `requirements-dev.txt`.
- The repo is bind-mounted into `/workspace`, so local datasets and checkpoints stay on the host.
- `stad_doctor` checks whether the Docker CLI and daemon are reachable before you try to build or run anything.
- The container startup now creates a named user matching your host UID/GID, so you should no longer see the shell prompt as `I have no name!`.
- To request GPU access on a Linux/NVIDIA machine, set `export STAD_DOCKER_GPU=1` before running `stad_run` or the helper commands.
