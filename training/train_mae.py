from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import build_dataloaders, build_dataset_report, load_localize_mi_config
from training.factory import build_mae, maybe_initialize_mae
from training.losses import masked_reconstruction_loss
from utils.checkpointing import save_checkpoint
from utils.logging import setup_logger, write_json
from utils.seed import seed_everything


def evaluate_mae(model, loader, device, mask_ratio: float) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            hr_eeg = batch["hr_eeg"].to(device)
            output = model(hr_eeg, mask_ratio=mask_ratio)
            total_loss += float(output.loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the DreamDiffusion-style EEG MAE.")
    parser.add_argument("--config", default="configs/stad_localize_mi_scale4.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_localize_mi_config(args.config)
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = args.output_dir
    if args.device is not None:
        config["train"]["device"] = args.device

    seed_everything(int(config["seed"]))
    output_dir = Path(config["experiment"]["output_dir"])
    logger = setup_logger("train_mae", output_dir)
    write_json(build_dataset_report(config["data"]["source_dir"], config["data"]["split"], config["data"]["scales"]), output_dir / "dataset_report.json")

    device = torch.device(config["train"]["device"])
    loaders = build_dataloaders(config)
    model = build_mae(config).to(device)
    maybe_initialize_mae(model, config, logger=logger)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    best_val_loss = float("inf")
    mask_ratio = float(config["train"]["mae_mask_ratio"])
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in loaders["train"]:
            hr_eeg = batch["hr_eeg"].to(device)
            output = model(hr_eeg, mask_ratio=mask_ratio)
            loss = masked_reconstruction_loss(output.loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        train_loss = running_loss / max(num_batches, 1)
        val_loss = evaluate_mae(model, loaders["val"], device, mask_ratio=mask_ratio)
        logger.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "best_val_loss": best_val_loss,
        }
        save_checkpoint(payload, config["paths"]["mae_checkpoint"])
        save_checkpoint(payload, output_dir / "mae_last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(payload, output_dir / "mae_best.pt")


if __name__ == "__main__":
    main()

