from data.localize_mi import LocalizeMIDataset, build_dataloaders, load_localize_mi_config


def test_localize_mi_dataset_shapes():
    config = load_localize_mi_config("configs/test_smoke.yaml")
    dataset = LocalizeMIDataset(
        source_dir=config["data"]["source_dir"],
        split="train",
        scale_factors=config["data"]["scales"],
        split_cfg=config["data"]["split"],
    )

    sample = dataset[0]
    assert sample["hr_eeg"].shape == (256, 1280)
    assert sample["lr_eeg"].shape == (64, 1280)
    assert sample["hr_positions"].shape == (256, 3)
    assert sample["lr_positions"].shape == (64, 3)
    assert sample["scale_factor"] == 4


def test_dataloaders_build_for_all_splits():
    config = load_localize_mi_config("configs/test_smoke.yaml")
    loaders = build_dataloaders(config)
    assert set(loaders.keys()) == {"train", "val", "test"}
    assert len(loaders["train"].dataset) > 0
    assert len(loaders["val"].dataset) > 0
    assert len(loaders["test"].dataset) > 0

