from __future__ import annotations

from pathlib import Path
from typing import Dict

from models import EEGMAE, GaussianDiffusion, MTD, STAD, STC
from utils.checkpointing import (
    best_matching_state_dict_variant,
    build_matching_state_dict,
    load_checkpoint,
    maybe_download_checkpoint,
    strip_module_prefix,
    unwrap_state_dict,
)

KNOWN_DREAMDIFFUSION_RELEASE_ID = "1Ygplxe1TB68-aYu082bjc89nD8Ngklnc"


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
    if (
        logger is not None
        and checkpoint_url
        and KNOWN_DREAMDIFFUSION_RELEASE_ID in checkpoint_url
        and not Path(checkpoint_path).exists()
    ):
        logger.warning(
            "The configured DreamDiffusion URL points to the README release checkpoint. It contains the released DreamDiffusion EEG encoder inside a larger generation checkpoint, so the download is large and the MAE decoder will still initialize randomly."
        )

    checkpoint = maybe_download_checkpoint(
        destination=checkpoint_path,
        url=checkpoint_url,
        overwrite=False,
    )
    payload = load_checkpoint(checkpoint)
    raw_state_dict = strip_module_prefix(unwrap_state_dict(payload))
    state_dict, prefix, raw_match_count = best_matching_state_dict_variant(mae, raw_state_dict)
    total = len(mae.state_dict())
    compatible, missing, unexpected, adapted = build_matching_state_dict(mae, state_dict)
    matched = len(compatible)
    coverage = matched / max(total, 1)
    if matched == 0:
        if logger is not None:
            logger.warning(
                "No compatible MAE tensors were found in %s after extracting the checkpoint state dict. Skipping DreamDiffusion initialization.",
                checkpoint,
            )
        return
    if coverage < 0.1:
        if logger is not None:
            logger.warning(
                "Checkpoint compatibility is too low to apply safely (%d/%d tensors matched, stripped_prefix=%r). Skipping DreamDiffusion initialization.",
                raw_match_count,
                total,
                prefix,
            )
        return
    mae.load_state_dict(compatible, strict=False)
    if logger is not None:
        decoder_missing = sum(
            1
            for key in missing
            if key.startswith("decoder_") or key.startswith("mask_token")
        )
        logger.info(
            "Loaded DreamDiffusion-style MAE weights from %s (matched=%d/%d, missing=%d, unexpected=%d, adapted=%d, stripped_prefix=%r, raw_match_count=%d)",
            checkpoint,
            matched,
            total,
            len(missing),
            len(unexpected),
            len(adapted),
            prefix,
            raw_match_count,
        )
        if decoder_missing:
            logger.info(
                "DreamDiffusion's public checkpoint initializes the MAE encoder path only; %d decoder-side tensors remain randomly initialized.",
                decoder_missing,
            )
        if coverage < 0.25:
            logger.warning(
                "Checkpoint compatibility is low (%.1f%% of tensors matched). DreamDiffusion's released weights likely target a different EEG shape/config than this 256x1280 MAE.",
                coverage * 100.0,
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
