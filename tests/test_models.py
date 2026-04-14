import torch

from training.factory import build_mae, build_stad
from data.localize_mi import load_localize_mi_config


def test_eeg_mae_forward_shapes():
    config = load_localize_mi_config("configs/test_smoke.yaml")
    mae = build_mae(config)
    eeg = torch.randn(2, 256, 1280)
    output = mae(eeg, mask_ratio=0.5)
    expected_full_tokens = config["model"]["mtd"]["sequence_length"]
    expected_kept_tokens = expected_full_tokens // 2

    assert output.reconstruction.shape == eeg.shape
    assert output.latent.shape == (2, expected_kept_tokens, 64)
    assert output.mask.shape == (2, expected_full_tokens)
    assert output.loss.ndim == 0


def test_stad_forward_and_reconstruction_shapes():
    config = load_localize_mi_config("configs/test_smoke.yaml")
    model = build_stad(config)
    hr_eeg = torch.randn(2, 256, 1280)
    lr_eeg = torch.randn(2, 64, 1280)
    lr_positions = torch.randn(2, 64, 3)

    output = model(lr_eeg, hr_eeg, lr_positions)
    reconstruction = model.reconstruct(lr_eeg, lr_positions, num_steps=2)

    assert output["loss"].ndim == 0
    assert output["latent"].shape == (2, 40, 64)
    assert output["condition_tokens"].shape == (2, 40, 64)
    assert output["pred_noise"].shape == (2, 40, 64)
    assert reconstruction.shape == (2, 256, 1280)
