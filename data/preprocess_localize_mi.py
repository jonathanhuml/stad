import argparse
from pathlib import Path

import torch

from data.localize_mi import build_dataset_report, load_localize_mi_config
from utils.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a Localize-MI dataset report and cached montage indices from the bundled .pt subject files."
    )
    parser.add_argument(
        "--config",
        default="configs/stad_localize_mi_scale4.yaml",
        help="Path to the STAD config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/localizeMI_processed",
        help="Directory where the dataset report and montage cache will be written.",
    )
    args = parser.parse_args()

    config = load_localize_mi_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_dataset_report(
        source_dir=config["data"]["source_dir"],
        split_cfg=config["data"]["split"],
        scales=config["data"]["scales"],
    )
    write_json(report, output_dir / "dataset_report.json")
    torch.save(report["subject_splits"], output_dir / "subject_splits.pt")


if __name__ == "__main__":
    main()
