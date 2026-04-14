from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import build_dataloaders, build_dataset_report, load_localize_mi_config
from training.factory import build_stad, load_model_checkpoint, maybe_initialize_mae
from utils.checkpointing import save_checkpoint
from utils.logging import setup_logger, write_json
from utils.seed import seed_everything


def evaluate_stad(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            lr_eeg = batch["lr_eeg"].to(device)
            hr_eeg = batch["hr_eeg"].to(device)
            lr_positions = batch["lr_positions"].to(device)
            output = model(lr_eeg, hr_eeg, lr_positions)
            total_loss += float(output["loss"].item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train STAD in the MAE latent space.")
    parser.add_argument("--config", default="configs/stad_localize_mi_scale4.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mae-checkpoint", default=None)
    args = parser.parse_args()

    config = load_localize_mi_config(args.config)
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = args.output_dir
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.mae_checkpoint is not None:
        config["paths"]["mae_checkpoint"] = args.mae_checkpoint

    seed_everything(int(config["seed"]))
    output_dir = Path(config["experiment"]["output_dir"])
    logger = setup_logger("train_stad", output_dir)
    write_json(build_dataset_report(config["data"]["source_dir"], config["data"]["split"], config["data"]["scales"]), output_dir / "dataset_report.json")

    device = torch.device(config["train"]["device"])
    loaders = build_dataloaders(config)
    model = build_stad(config).to(device)

    mae_checkpoint = Path(config["paths"]["mae_checkpoint"])
    if mae_checkpoint.exists():
        load_model_checkpoint(model.mae, mae_checkpoint)
        logger.info("Loaded MAE checkpoint from %s", mae_checkpoint)
    else:
        maybe_initialize_mae(model.mae, config, logger=logger)
        logger.info("MAE checkpoint %s was not found, using DreamDiffusion-style initialization only.", mae_checkpoint)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    best_val_loss = float("inf")
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in loaders["train"]:
            lr_eeg = batch["lr_eeg"].to(device)
            hr_eeg = batch["hr_eeg"].to(device)
            lr_positions = batch["lr_positions"].to(device)
            output = model(lr_eeg, hr_eeg, lr_positions)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, float(config["train"]["grad_clip"]))
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        train_loss = running_loss / max(num_batches, 1)
        val_loss = evaluate_stad(model, loaders["val"], device)
        logger.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "best_val_loss": best_val_loss,
        }
        save_checkpoint(payload, config["paths"]["stad_checkpoint"])
        save_checkpoint(payload, output_dir / "stad_last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(payload, output_dir / "stad_best.pt")


if __name__ == "__main__":
    main()

