from __future__ import annotations

import math
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset

from data.montages import MontageLibrary, build_nested_montages


def load_localize_mi_config(config_path: str | Path) -> Dict:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def _load_subject_file(subject_path: str) -> Dict:
    return torch.load(subject_path, map_location="cpu")


def _parse_subject_id(payload: Dict, path: Path) -> str:
    metadata = payload.get("metadata", {})
    subject_id = metadata.get("subject_id")
    if subject_id is not None:
        return f"{int(subject_id):02d}"
    return path.stem.split("_")[1]


def _discover_subject_files(source_dir: str | Path) -> List[Path]:
    files = sorted(Path(source_dir).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No subject .pt files found under {source_dir}")
    return files


def _compute_split_counts(num_subjects: int, ratios: Iterable[float]) -> tuple[int, int, int]:
    train_ratio, val_ratio, test_ratio = ratios
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-4):
        raise ValueError("Split ratios must sum to 1.0")

    train_count = int(num_subjects * train_ratio)
    val_count = int(num_subjects * val_ratio)
    test_count = num_subjects - train_count - val_count

    if num_subjects >= 3:
        if val_count == 0:
            val_count = 1
            train_count = max(1, train_count - 1)
        if test_count == 0:
            test_count = 1
            train_count = max(1, train_count - 1)

    while train_count + val_count + test_count < num_subjects:
        train_count += 1
    while train_count + val_count + test_count > num_subjects:
        if train_count > max(1, val_count, test_count):
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            test_count -= 1

    return train_count, val_count, test_count


def _split_subjects(subject_ids: List[str], split_cfg: Dict) -> Dict[str, set[str]]:
    shuffled_ids = list(subject_ids)
    random.Random(int(split_cfg["seed"])).shuffle(shuffled_ids)
    counts = _compute_split_counts(len(shuffled_ids), split_cfg["ratios"])
    train_count, val_count, _ = counts

    train_subjects = set(shuffled_ids[:train_count])
    val_subjects = set(shuffled_ids[train_count : train_count + val_count])
    test_subjects = set(shuffled_ids[train_count + val_count :])
    return {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }


@dataclass
class DatasetDeviation:
    name: str
    detail: str


class LocalizeMIDataset(Dataset):
    def __init__(
        self,
        source_dir: str | Path,
        split: str,
        scale_factors: Iterable[int],
        split_cfg: Dict,
        batch_size_hint: int | None = None,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.split = split
        self.scale_factors = [int(scale) for scale in scale_factors]
        self.batch_size_hint = batch_size_hint

        subject_files = _discover_subject_files(self.source_dir)
        subject_payloads = [_load_subject_file(str(path)) for path in subject_files]
        subject_ids = [_parse_subject_id(payload, path) for payload, path in zip(subject_payloads, subject_files)]
        self.subject_splits = _split_subjects(subject_ids, split_cfg)

        first_payload = subject_payloads[0]
        first_positions = first_payload["channel_positions"][0]
        full_count = int(first_positions.shape[0])
        expected_counts = [full_count // scale for scale in self.scale_factors]
        self.montages: MontageLibrary = build_nested_montages(
            positions=first_positions,
            channel_counts=[full_count] + expected_counts,
        )

        self.index: list[dict] = []
        self.deviations = self._extract_deviations(first_payload)
        for subject_path, payload, subject_id in zip(subject_files, subject_payloads, subject_ids):
            if subject_id not in self.subject_splits[split]:
                continue
            num_epochs = len(payload["data"])
            for epoch_index in range(num_epochs):
                for scale_factor in self.scale_factors:
                    self.index.append(
                        {
                            "subject_path": str(subject_path),
                            "subject_id": subject_id,
                            "epoch_index": epoch_index,
                            "scale_factor": scale_factor,
                        }
                    )

    def _extract_deviations(self, payload: Dict) -> list[DatasetDeviation]:
        metadata = payload.get("metadata", {})
        preprocessing = metadata.get("preprocessing", {})
        deviations = [
            DatasetDeviation(
                name="sampling_rate",
                detail=f"Bundled data is sampled at {metadata.get('sampling_rate')} Hz, not the 8000 Hz acquisition rate described in the paper.",
            ),
            DatasetDeviation(
                name="epoch_length",
                detail=f"Bundled epochs have {metadata.get('n_timepoints')} samples, which corresponds to {metadata.get('n_timepoints') / metadata.get('sampling_rate'):.2f} seconds at the provided sampling rate.",
            ),
            DatasetDeviation(
                name="preprocessed_input",
                detail="The workspace only contains already-preprocessed subject files, so raw filtering and epoch extraction cannot be reproduced from source recordings here.",
            ),
        ]
        if preprocessing.get("resampled"):
            deviations.append(
                DatasetDeviation(
                    name="resampled",
                    detail="Metadata marks the bundled EEG as resampled before this pipeline.",
                )
            )
        if preprocessing.get("z_score_normalized"):
            deviations.append(
                DatasetDeviation(
                    name="normalized",
                    detail="Metadata marks the bundled EEG as already z-score normalized.",
                )
            )
        if not preprocessing.get("notch_frequencies"):
            deviations.append(
                DatasetDeviation(
                    name="notch_metadata",
                    detail="No notch frequencies are recorded in the bundled metadata, so the paper's reported 50/100/150/200 Hz notch chain cannot be verified here.",
                )
            )
        return deviations

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor | int | str]:
        record = self.index[item]
        payload = _load_subject_file(record["subject_path"])
        epoch_index = int(record["epoch_index"])
        scale_factor = int(record["scale_factor"])
        montage_indices = self.montages.indices_for_scale(scale_factor)

        hr_eeg = payload["data"][epoch_index].float()
        hr_positions = payload["channel_positions"][epoch_index].float()
        lr_eeg = hr_eeg[montage_indices]
        lr_positions = hr_positions[montage_indices]
        label = int(payload["labels"][epoch_index].item())

        return {
            "hr_eeg": hr_eeg,
            "lr_eeg": lr_eeg,
            "hr_positions": hr_positions,
            "lr_positions": lr_positions,
            "label": label,
            "scale_factor": scale_factor,
            "subject_id": record["subject_id"],
            "epoch_index": epoch_index,
        }


def build_dataset_report(source_dir: str | Path, split_cfg: Dict, scales: Iterable[int]) -> Dict:
    subject_files = _discover_subject_files(source_dir)
    payloads = [_load_subject_file(str(path)) for path in subject_files]
    subject_ids = [_parse_subject_id(payload, path) for payload, path in zip(payloads, subject_files)]
    subject_splits = _split_subjects(subject_ids, split_cfg)

    total_epochs = sum(len(payload["data"]) for payload in payloads)
    first_payload = payloads[0]
    first_epoch = first_payload["data"][0]
    first_positions = first_payload["channel_positions"][0]

    dataset = LocalizeMIDataset(
        source_dir=source_dir,
        split="train",
        scale_factors=scales,
        split_cfg=split_cfg,
    )

    return {
        "source_dir": str(source_dir),
        "num_subject_files": len(subject_files),
        "subject_ids": subject_ids,
        "subject_splits": {name: sorted(values) for name, values in subject_splits.items()},
        "total_epochs": total_epochs,
        "epoch_shape": list(first_epoch.shape),
        "channel_position_shape": list(first_positions.shape),
        "sampling_rate_hz": first_payload["metadata"].get("sampling_rate"),
        "available_scales": list(scales),
        "available_montage_sizes": sorted(dataset.montages.indices_by_count.keys(), reverse=True),
        "deviations": [
            {"name": deviation.name, "detail": deviation.detail}
            for deviation in dataset.deviations
        ],
    }


def build_dataloaders(config: Dict) -> Dict[str, DataLoader]:
    source_dir = config["data"]["source_dir"]
    split_cfg = config["data"]["split"]
    scales = config["data"]["scales"]
    loaders: Dict[str, DataLoader] = {}

    for split_name, shuffle in (("train", True), ("val", False), ("test", False)):
        dataset = LocalizeMIDataset(
            source_dir=source_dir,
            split=split_name,
            scale_factors=scales,
            split_cfg=split_cfg,
            batch_size_hint=config["train"]["batch_size"],
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"] if split_name == "train" else config["eval"]["batch_size"],
            shuffle=shuffle and len(dataset) > 0,
            num_workers=config["data"]["num_workers"],
            pin_memory=bool(config["train"]["device"].startswith("cuda")),
        )
    return loaders
