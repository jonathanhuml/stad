from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.request import urlretrieve

import torch
from torch import nn


COMMON_STATE_DICT_KEYS = (
    "state_dict",
    "model",
    "module",
    "net",
    "ema",
)


def unwrap_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in COMMON_STATE_DICT_KEYS:
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return unwrap_state_dict(value)
        if payload and all(isinstance(key, str) for key in payload.keys()):
            return payload
    raise ValueError("Could not locate a state_dict in the provided checkpoint payload.")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def save_checkpoint(payload: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_matching_state_dict(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[list[str], list[str]]:
    current = module.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    missing = [key for key in current.keys() if key not in compatible]
    unexpected = [key for key in state_dict.keys() if key not in compatible]
    module.load_state_dict(compatible, strict=False)
    return missing, unexpected


def maybe_download_checkpoint(
    destination: str | Path,
    url: str | None,
    overwrite: bool = False,
) -> Path:
    path = Path(destination)
    if path.exists() and not overwrite:
        return path
    if url is None:
        raise FileNotFoundError(f"{path} does not exist and no download URL was provided.")
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)
    return path

