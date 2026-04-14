import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from urllib.parse import parse_qs, urlparse

import requests
import torch
import torch.nn.functional as F
from torch import nn


COMMON_STATE_DICT_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "module",
    "net",
    "ema",
)


def _is_tensor_state_dict(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and bool(payload)
        and all(isinstance(key, str) for key in payload.keys())
        and all(isinstance(value, torch.Tensor) for value in payload.values())
    )


def unwrap_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in COMMON_STATE_DICT_KEYS:
            value = payload.get(key)
            if isinstance(value, dict) and value:
                try:
                    return unwrap_state_dict(value)
                except ValueError:
                    continue
        if _is_tensor_state_dict(payload):
            return dict(payload)
        for value in payload.values():
            if isinstance(value, dict) and value:
                try:
                    return unwrap_state_dict(value)
                except ValueError:
                    continue
    raise ValueError("Could not locate a state_dict in the provided checkpoint payload.")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def _iter_prefix_variants(keys: Iterable[str], max_depth: int = 4) -> list[str]:
    prefixes = {""}
    for key in keys:
        parts = key.split(".")
        for depth in range(1, min(max_depth, len(parts) - 1) + 1):
            prefixes.add(".".join(parts[:depth]) + ".")
    return sorted(prefixes, key=len)


def _strip_key_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return dict(state_dict)
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix) :]] = value
    return stripped


def best_matching_state_dict_variant(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], str, int]:
    current = module.state_dict()
    best_variant = dict(state_dict)
    best_prefix = ""
    best_match_count = -1

    for prefix in _iter_prefix_variants(state_dict.keys()):
        variant = _strip_key_prefix(state_dict, prefix)
        match_count = sum(
            1
            for key, value in variant.items()
            if key in current and current[key].shape == value.shape
        )
        if match_count > best_match_count:
            best_variant = variant
            best_prefix = prefix
            best_match_count = match_count

    return best_variant, best_prefix, best_match_count


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def save_checkpoint(payload: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_matching_state_dict(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[list[str], list[str], int, list[str]]:
    compatible, missing, unexpected, adapted = build_matching_state_dict(module, state_dict)
    module.load_state_dict(compatible, strict=False)
    return missing, unexpected, len(compatible), adapted


def build_matching_state_dict(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], list[str], list[str], list[str]]:
    current = module.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    adapted: list[str] = []
    for key, value in state_dict.items():
        if key not in current:
            continue
        target = current[key]
        if target.shape == value.shape:
            compatible[key] = value
            continue

        resized = _adapt_tensor_to_target(key, value, target)
        if resized is not None:
            compatible[key] = resized
            adapted.append(key)

    missing = [key for key in current.keys() if key not in compatible]
    unexpected = [key for key in state_dict.keys() if key not in compatible]
    return compatible, missing, unexpected, adapted


def _adapt_tensor_to_target(
    key: str,
    value: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    if value.shape == target.shape:
        return value
    if key.endswith("pos_embed") and value.ndim == 3 and target.ndim == 3 and value.shape[0] == target.shape[0] == 1 and value.shape[2] == target.shape[2]:
        cls_tokens = value[:, :1]
        token_body = value[:, 1:, :]
        resized_body = F.interpolate(
            token_body.transpose(1, 2),
            size=target.shape[1] - 1,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        resized = torch.cat((cls_tokens, resized_body), dim=1)
        return resized.to(dtype=target.dtype)
    if key.endswith("patch_embed.proj.weight") and value.ndim == 3 and target.ndim == 3 and value.shape[0] == target.shape[0]:
        resized = F.interpolate(
            value.unsqueeze(1),
            size=(target.shape[1], target.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return resized.to(dtype=target.dtype)
    return None


def _google_drive_file_id(url: str) -> str | None:
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return None

    path_parts = [part for part in parsed.path.split("/") if part]
    if "file" in path_parts and "d" in path_parts:
        d_index = path_parts.index("d")
        if d_index + 1 < len(path_parts):
            return path_parts[d_index + 1]

    query_id = parse_qs(parsed.query).get("id")
    if query_id:
        return query_id[0]
    return None


def _download_google_drive_file(destination: Path, url: str) -> Path:
    file_id = _google_drive_file_id(url)
    if file_id is None:
        raise ValueError(f"Could not extract a Google Drive file id from {url}")

    session = requests.Session()
    params = {"id": file_id, "export": "download"}
    response = session.get("https://drive.google.com/uc", params=params, stream=True)
    response.raise_for_status()

    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    text = response.text if "text/html" in response.headers.get("Content-Type", "") else ""
    if confirm_token is None:
        match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
        if match:
            confirm_token = match.group(1)

    if "download-form" in text:
        action_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', text)
        inputs = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', text))
        if action_match and inputs:
            response.close()
            response = session.get(action_match.group(1), params=inputs, stream=True)
            response.raise_for_status()
            text = ""

    elif confirm_token is not None:
        response.close()
        response = session.get(
            "https://drive.google.com/uc",
            params={"id": file_id, "export": "download", "confirm": confirm_token},
            stream=True,
        )
        response.raise_for_status()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return destination


def _download_regular_file(destination: Path, url: str) -> Path:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return destination


def maybe_download_checkpoint(
    destination: str | Path,
    url: str | None,
    overwrite: bool = False,
) -> Path:
    path = Path(destination)
    if path.exists() and not overwrite:
        try:
            header = path.read_bytes()[:256].lstrip()
            if not (header.startswith(b"<!DOCTYPE html") or header.startswith(b"<html")):
                return path
        except OSError:
            pass
    if url is None:
        raise FileNotFoundError(f"{path} does not exist and no download URL was provided.")
    if _google_drive_file_id(url) is not None:
        return _download_google_drive_file(path, url)
    return _download_regular_file(path, url)
