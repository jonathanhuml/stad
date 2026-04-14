import torch


def normalize_positions(positions: torch.Tensor) -> torch.Tensor:
    centered = positions - positions.mean(dim=0, keepdim=True)
    max_norm = centered.norm(dim=-1).amax().clamp_min(1e-6)
    return centered / max_norm


def squared_distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    diff = positions[:, None, :] - positions[None, :, :]
    return (diff * diff).sum(dim=-1)

