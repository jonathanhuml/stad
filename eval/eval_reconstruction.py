from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import build_dataloaders, load_localize_mi_config
from training.factory import build_stad, load_model_checkpoint, maybe_initialize_mae
from training.metrics import reconstruction_metrics
from utils.logging import setup_logger, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate STAD EEG reconstruction quality.")
    parser.add_argument("--config", default="configs/stad_localize_mi_scale4.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--stad-checkpoint", default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_localize_mi_config(args.config)
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.stad_checkpoint is not None:
        config["paths"]["stad_checkpoint"] = args.stad_checkpoint

    device = torch.device(config["train"]["device"])
    logger = setup_logger("eval_reconstruction", config["experiment"]["output_dir"])
    loaders = build_dataloaders(config)
    model = build_stad(config).to(device)

    mae_checkpoint = Path(config["paths"]["mae_checkpoint"])
    if mae_checkpoint.exists():
        load_model_checkpoint(model.mae, mae_checkpoint)
    else:
        maybe_initialize_mae(model.mae, config, logger=logger)

    load_model_checkpoint(model, config["paths"]["stad_checkpoint"])
    model.eval()

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in loaders["test"]:
            lr_eeg = batch["lr_eeg"].to(device)
            hr_eeg = batch["hr_eeg"].to(device)
            lr_positions = batch["lr_positions"].to(device)
            pred = model.reconstruct(lr_eeg, lr_positions, num_steps=args.sample_steps or config["eval"]["sample_steps"])
            all_predictions.append(pred.cpu())
            all_targets.append(hr_eeg.cpu())

    metrics = reconstruction_metrics(torch.cat(all_predictions, dim=0), torch.cat(all_targets, dim=0))
    logger.info("reconstruction_metrics=%s", metrics)
    write_json(metrics, Path(config["experiment"]["output_dir"]) / "reconstruction_metrics.json")


if __name__ == "__main__":
    main()

