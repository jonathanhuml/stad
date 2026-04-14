from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from data import build_dataloaders, load_localize_mi_config
from training.factory import build_mae, maybe_initialize_mae
from training.metrics import reconstruction_metrics
from utils.logging import setup_logger
from utils.seed import seed_everything


@torch.no_grad()
def evaluate_subset(model, loader, device, mask_ratio: float) -> dict[str, float]:
    model.eval()
    masked_losses: list[float] = []
    targets: list[torch.Tensor] = []
    reconstructions: list[torch.Tensor] = []
    for batch in loader:
        hr_eeg = batch["hr_eeg"].to(device)
        masked_output = model(hr_eeg, mask_ratio=mask_ratio)
        recon_output = model(hr_eeg, mask_ratio=0.0)
        masked_losses.append(float(masked_output.loss.item()))
        targets.append(hr_eeg.cpu())
        reconstructions.append(recon_output.reconstruction.cpu())

    target = torch.cat(targets, dim=0)
    prediction = torch.cat(reconstructions, dim=0)
    metrics = reconstruction_metrics(prediction, target)
    metrics["masked_loss"] = sum(masked_losses) / max(len(masked_losses), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit the MAE on a tiny fixed train subset.")
    parser.add_argument("--config", default="configs/stad_localize_mi_scale4.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--subset-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    args = parser.parse_args()

    config = load_localize_mi_config(args.config)
    if args.device is not None:
        config["train"]["device"] = args.device

    seed_everything(int(config["seed"]))
    logger = setup_logger("overfit_mae", config["experiment"]["output_dir"])

    device = torch.device(config["train"]["device"])
    loaders = build_dataloaders(config)
    train_dataset = loaders["train"].dataset
    subset_size = min(max(1, int(args.subset_size)), len(train_dataset))
    subset = Subset(train_dataset, list(range(subset_size)))
    loader = DataLoader(
        subset,
        batch_size=min(int(config["train"]["batch_size"]), subset_size),
        shuffle=False,
        num_workers=0,
    )

    model = build_mae(config).to(device)
    maybe_initialize_mae(model, config, logger=logger)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"] if args.lr is None else args.lr),
        weight_decay=float(config["train"]["weight_decay"] if args.weight_decay is None else args.weight_decay),
    )
    mask_ratio = float(config["train"]["mae_mask_ratio"] if args.mask_ratio is None else args.mask_ratio)
    total_steps = max(1, int(args.steps))
    log_every = max(1, int(args.log_every))

    metrics = evaluate_subset(model, loader, device, mask_ratio)
    logger.info(
        "step=0 subset_size=%d masked_loss=%.6f pcc=%.6f nmse=%.6f snr_db=%.6f mae=%.6f",
        subset_size,
        metrics["masked_loss"],
        metrics["pcc"],
        metrics["nmse"],
        metrics["snr_db"],
        metrics["mae"],
    )

    step = 0
    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break
            step += 1
            hr_eeg = batch["hr_eeg"].to(device)
            output = model(hr_eeg, mask_ratio=mask_ratio)
            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
            optimizer.step()

            if step % log_every == 0 or step == total_steps:
                metrics = evaluate_subset(model, loader, device, mask_ratio)
                logger.info(
                    "step=%d subset_size=%d masked_loss=%.6f pcc=%.6f nmse=%.6f snr_db=%.6f mae=%.6f",
                    step,
                    subset_size,
                    metrics["masked_loss"],
                    metrics["pcc"],
                    metrics["nmse"],
                    metrics["snr_db"],
                    metrics["mae"],
                )


if __name__ == "__main__":
    main()
