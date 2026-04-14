from __future__ import annotations

import torch
from torch import nn

from models.common import Mlp, TimestepEmbedding, get_1d_sincos_pos_embed


class MultiScaleConvMixer(nn.Module):
    def __init__(self, d_model: int, kernels: list[int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2),
                    nn.BatchNorm1d(d_model),
                    nn.GELU(),
                )
                for kernel in kernels
            ]
        )
        self.proj = nn.Conv1d(d_model * len(kernels), d_model, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.transpose(1, 2)
        outputs = [branch(x) for branch in self.branches]
        x = torch.cat(outputs, dim=1)
        x = self.proj(x)
        return x.transpose(1, 2)


class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = Mlp(d_model, int(d_model * mlp_ratio), drop=dropout)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(
        self,
        latent_tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.norm1(latent_tokens)
        attn_output, _ = self.self_attn(hidden, hidden, hidden, need_weights=False)
        latent_tokens = latent_tokens + attn_output

        cross_input = self.norm2(latent_tokens)
        cross_output, _ = self.cross_attn(
            cross_input,
            condition_tokens,
            condition_tokens,
            need_weights=False,
        )
        latent_tokens = latent_tokens + cross_output

        feedforward_input = self.norm3(latent_tokens) + self.time_proj(time_embedding).unsqueeze(1)
        latent_tokens = latent_tokens + self.mlp(feedforward_input)
        return latent_tokens


class MTD(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        d_model: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        conv_kernels: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        conv_kernels = conv_kernels or [3, 5, 7, 9]

        self.time_embedding = TimestepEmbedding(d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.token_mixer = MultiScaleConvMixer(d_model, conv_kernels)
        self.register_buffer(
            "token_pos_embed",
            get_1d_sincos_pos_embed(d_model, sequence_length, cls_token=False),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_latent.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected {self.sequence_length} latent tokens, got {noisy_latent.shape[1]}"
            )

        time_embedding = self.time_embedding(timesteps)
        tokens = self.input_norm(noisy_latent)
        tokens = tokens + self.token_pos_embed[:, : noisy_latent.shape[1]]
        tokens = tokens + time_embedding.unsqueeze(1)
        tokens = tokens + self.token_mixer(tokens)

        for block in self.blocks:
            tokens = block(tokens, condition_tokens, time_embedding)

        tokens = self.norm(tokens)
        return self.output_proj(tokens)

