import torch

from models.eeg_mae import EEGMAE
from utils.checkpointing import (
    _google_drive_file_id,
    best_matching_state_dict_variant,
    load_matching_state_dict,
    unwrap_state_dict,
)


def test_google_drive_file_id_extraction():
    url = "https://drive.google.com/file/d/1Ygplxe1TB68-aYu082bjc89nD8Ngklnc/view?usp=drive_link"
    assert _google_drive_file_id(url) == "1Ygplxe1TB68-aYu082bjc89nD8Ngklnc"


def test_best_matching_state_dict_variant_strips_prefix():
    module = EEGMAE(
        in_chans=256,
        time_steps=1280,
        patch_size=32,
        embed_dim=64,
        depth=2,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2.0,
    )
    state_dict = module.state_dict()
    prefixed = {f"encoder.{key}": value.clone() for key, value in state_dict.items()}

    variant, prefix, matched = best_matching_state_dict_variant(module, prefixed)
    assert prefix == "encoder."
    assert matched == len(state_dict)

    missing, unexpected, loaded, adapted = load_matching_state_dict(module, variant)
    assert loaded == len(state_dict)
    assert missing == []
    assert unexpected == []
    assert adapted == []


def test_unwrap_state_dict_prefers_nested_model_state_dict():
    nested = {"encoder.weight": torch.randn(2, 2), "encoder.bias": torch.randn(2)}
    payload = {
        "model_state_dict": nested,
        "config": object(),
        "state": {"epoch": 1},
    }

    unwrapped = unwrap_state_dict(payload)
    assert unwrapped.keys() == nested.keys()
    for key, value in nested.items():
        assert torch.equal(unwrapped[key], value)


def test_load_matching_state_dict_adapts_positional_and_patch_embeddings():
    module = EEGMAE(
        in_chans=256,
        time_steps=1280,
        patch_size=16,
        embed_dim=1024,
        depth=2,
        num_heads=16,
        decoder_embed_dim=128,
        decoder_depth=1,
        decoder_num_heads=4,
        encoder_mlp_ratio=1.0,
        decoder_mlp_ratio=2.0,
    )
    state_dict = {
        "cls_token": torch.randn(1, 1, 1024),
        "pos_embed": torch.randn(1, 129, 1024),
        "patch_embed.proj.weight": torch.randn(1024, 128, 4),
        "patch_embed.proj.bias": torch.randn(1024),
        "norm.weight": torch.randn(1024),
        "norm.bias": torch.randn(1024),
    }

    missing, unexpected, loaded, adapted = load_matching_state_dict(module, state_dict)
    assert "pos_embed" in adapted
    assert "patch_embed.proj.weight" in adapted
    assert loaded == len(state_dict)
    assert "patch_embed.proj.weight" not in missing
    assert unexpected == []
