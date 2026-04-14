from __future__ import annotations

from pathlib import Path
from typing import Dict

from models import EEGMAE, GaussianDiffusion, MTD, STAD, STC
from utils.checkpointing import (
    load_checkpoint,
    load_matching_state_dict,
    maybe_download_checkpoint,
    strip_module_prefix,
    unwrap_state_dict,
)


def build_mae(config: Dict) -> EEGMAE:
    mae_cfg = dict(config["model"]["mae"])
    mae_cfg.pop("pretrained_checkpoint_path", None)
    mae_cfg.pop("pretrained_checkpoint_url", None)
    return EEGMAE(**mae_cfg)


def maybe_initialize_mae(mae: EEGMAE, config: Dict, logger=None) -> None:
    mae_cfg = config["model"]["mae"]
    checkpoint_path = mae_cfg.get("pretrained_checkpoint_path")
    checkpoint_url = mae_cfg.get("pretrained_checkpoint_url")
    if not checkpoint_path:
        return
    if not Path(checkpoint_path).exists() and not checkpoint_url:
        if logger is not None:
            logger.info(
                "No DreamDiffusion checkpoint was found at %s and no download URL is configured.",
                checkpoint_path,
            )
        return

    checkpoint = maybe_download_checkpoint(
        destination=checkpoint_path,
        url=checkpoint_url,
        overwrite=False,
    )
    payload = load_checkpoint(checkpoint)
    state_dict = strip_module_prefix(unwrap_state_dict(payload))
    missing, unexpected = load_matching_state_dict(mae, state_dict)
    if logger is not None:
        logger.info(
            "Loaded DreamDiffusion-style MAE weights from %s (missing=%d, unexpected=%d)",
            checkpoint,
            len(missing),
            len(unexpected),
        )


def build_stad(config: Dict) -> STAD:
    mae = build_mae(config)
    stc = STC(**config["model"]["stc"])
    mtd = MTD(**config["model"]["mtd"])
    diffusion = GaussianDiffusion(**config["model"]["diffusion"])
    return STAD(
        mae=mae,
        stc=stc,
        mtd=mtd,
        diffusion=diffusion,
        freeze_mae=bool(config["train"]["freeze_mae"]),
    )


def load_model_checkpoint(model, checkpoint_path: str | Path) -> Dict:
    payload = load_checkpoint(checkpoint_path)
    state_dict = strip_module_prefix(unwrap_state_dict(payload))
    model.load_state_dict(state_dict, strict=False)
    return payload
