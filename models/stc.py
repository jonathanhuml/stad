from __future__ import annotations

import torch
from torch import nn

from models.common import Block, get_1d_sincos_pos_embed


class STC(nn.Module):
    def __init__(
        self,
        patch_size: int,
        time_steps: int,
        d_model: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        position_dim: int = 3,
    ) -> None:
        super().__init__()
        if time_steps % patch_size != 0:
            raise ValueError("time_steps must be divisible by patch_size")

        self.patch_size = patch_size
        self.time_steps = time_steps
        self.num_tokens = time_steps // patch_size
        self.d_model = d_model

        self.position_mlp = nn.Sequential(
            nn.Linear(position_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.channel_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.register_buffer(
            "time_pos_embed",
            get_1d_sincos_pos_embed(d_model, self.num_tokens, cls_token=False),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [Block(d_model, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, lr_eeg: torch.Tensor, lr_positions: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time_steps = lr_eeg.shape
        if time_steps != self.time_steps:
            raise ValueError(
                f"Expected LR EEG with {self.time_steps} time steps, received {time_steps}"
            )

        tokens = self.temporal_encoder(lr_eeg.reshape(batch_size * channels, 1, time_steps))
        tokens = tokens.transpose(1, 2).reshape(batch_size, channels, self.num_tokens, self.d_model)

        position_features = self.position_mlp(lr_positions).unsqueeze(2)
        tokens = tokens + position_features

        weights = torch.softmax(self.channel_pool(tokens), dim=1)
        tokens = (weights * tokens).sum(dim=1)
        tokens = tokens + self.time_pos_embed[:, : self.num_tokens]

        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

