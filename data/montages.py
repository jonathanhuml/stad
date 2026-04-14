from dataclasses import dataclass
from typing import Dict, Iterable

import torch

from utils.positions import normalize_positions


def _farthest_point_subset(positions: torch.Tensor, subset_size: int) -> torch.Tensor:
    if subset_size > positions.shape[0]:
        raise ValueError("subset_size cannot exceed the number of input positions.")

    positions = normalize_positions(positions.float())
    centroid = positions.mean(dim=0, keepdim=True)
    first_index = int(((positions - centroid) ** 2).sum(dim=-1).argmax().item())

    chosen = [first_index]
    min_distances = ((positions - positions[first_index]) ** 2).sum(dim=-1)

    while len(chosen) < subset_size:
        next_index = int(min_distances.argmax().item())
        chosen.append(next_index)
        candidate_distances = ((positions - positions[next_index]) ** 2).sum(dim=-1)
        min_distances = torch.minimum(min_distances, candidate_distances)

    chosen = torch.tensor(chosen, dtype=torch.long)
    return chosen.sort().values


@dataclass
class MontageLibrary:
    full_count: int
    indices_by_count: Dict[int, torch.Tensor]

    def indices_for_scale(self, scale_factor: int) -> torch.Tensor:
        target_count = self.full_count // scale_factor
        if target_count not in self.indices_by_count:
            available = sorted(self.indices_by_count.keys(), reverse=True)
            raise KeyError(
                f"Scale {scale_factor} is not available. Known montage sizes: {available}"
            )
        return self.indices_by_count[target_count]


def build_nested_montages(
    positions: torch.Tensor,
    channel_counts: Iterable[int],
) -> MontageLibrary:
    channel_counts = sorted(set(int(count) for count in channel_counts), reverse=True)
    full_count = int(positions.shape[0])
    if channel_counts[0] != full_count:
        channel_counts = [full_count] + channel_counts

    indices_by_count: Dict[int, torch.Tensor] = {full_count: torch.arange(full_count)}
    current_indices = indices_by_count[full_count]

    for next_count in channel_counts[1:]:
        subset_positions = positions[current_indices]
        nested_indices = _farthest_point_subset(subset_positions, next_count)
        current_indices = current_indices[nested_indices]
        indices_by_count[next_count] = current_indices

    return MontageLibrary(full_count=full_count, indices_by_count=indices_by_count)

