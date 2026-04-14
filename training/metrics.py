from __future__ import annotations

import torch


def _flatten_channels_time(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1)


def pearson_correlation(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _flatten_channels_time(prediction)
    truth = _flatten_channels_time(target)
    pred = pred - pred.mean(dim=1, keepdim=True)
    truth = truth - truth.mean(dim=1, keepdim=True)
    numerator = (pred * truth).sum(dim=1)
    denominator = pred.norm(dim=1) * truth.norm(dim=1)
    return numerator / denominator.clamp_min(1.0e-8)


def nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction - target).pow(2).sum(dim=(1, 2))
    denominator = target.pow(2).sum(dim=(1, 2)).clamp_min(1.0e-8)
    return numerator / denominator


def snr_db(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    signal_power = target.pow(2).mean(dim=(1, 2)).clamp_min(1.0e-8)
    noise_power = (target - prediction).pow(2).mean(dim=(1, 2)).clamp_min(1.0e-8)
    return 10.0 * torch.log10(signal_power / noise_power)


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).abs().mean(dim=(1, 2))


def reconstruction_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "pcc": float(pearson_correlation(prediction, target).mean().item()),
        "nmse": float(nmse(prediction, target).mean().item()),
        "snr_db": float(snr_db(prediction, target).mean().item()),
        "mae": float(mae(prediction, target).mean().item()),
    }

