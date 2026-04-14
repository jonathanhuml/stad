from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.common import Block, PatchEmbed1D, get_1d_sincos_pos_embed


@dataclass
class EEGMAEOutput:
    loss: torch.Tensor
    reconstruction: torch.Tensor
    latent: torch.Tensor
    mask: torch.Tensor


class EEGMAE(nn.Module):
    def __init__(
        self,
        in_chans: int = 256,
        time_steps: int = 1280,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        encoder_mlp_ratio: float | None = None,
        decoder_mlp_ratio: float | None = None,
        norm_pix_loss: bool = False,
    ) -> None:
        super().__init__()
        if time_steps % patch_size != 0:
            raise ValueError("time_steps must be divisible by patch_size")

        self.in_chans = in_chans
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.norm_pix_loss = norm_pix_loss
        self.num_patches = time_steps // patch_size
        encoder_mlp_ratio = float(mlp_ratio if encoder_mlp_ratio is None else encoder_mlp_ratio)
        decoder_mlp_ratio = float(mlp_ratio if decoder_mlp_ratio is None else decoder_mlp_ratio)

        self.patch_embed = PatchEmbed1D(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            "pos_embed",
            get_1d_sincos_pos_embed(embed_dim, self.num_patches, cls_token=True),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio=encoder_mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer(
            "decoder_pos_embed",
            get_1d_sincos_pos_embed(decoder_embed_dim, self.num_patches, cls_token=True),
            persistent=True,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=decoder_mlp_ratio)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_size)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def patchify(self, signals: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time_steps = signals.shape
        if channels != self.in_chans or time_steps != self.time_steps:
            raise ValueError(
                f"Expected input of shape [B, {self.in_chans}, {self.time_steps}], "
                f"received {tuple(signals.shape)}"
            )
        patches = signals.view(batch_size, channels, self.num_patches, self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(
            batch_size, self.num_patches, channels * self.patch_size
        )
        return patches

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, length, dim = patches.shape
        if length != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches, received {length}"
            )
        signals = patches.view(batch_size, length, self.in_chans, self.patch_size)
        signals = signals.permute(0, 2, 1, 3).reshape(
            batch_size, self.in_chans, self.time_steps
        )
        return signals

    def random_masking(
        self,
        tokens: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, length, dim = tokens.shape
        keep_length = int(length * (1.0 - mask_ratio))
        noise = torch.rand(batch_size, length, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :keep_length]
        tokens_kept = torch.gather(
            tokens,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, dim),
        )

        mask = torch.ones(batch_size, length, device=tokens.device)
        mask[:, :keep_length] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return tokens_kept, mask, ids_restore

    def forward_encoder(
        self,
        signals: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(signals)
        tokens = tokens + self.pos_embed[:, 1 : self.num_patches + 1]

        if mask_ratio > 0:
            tokens, mask, ids_restore = self.random_masking(tokens, mask_ratio)
        else:
            batch_size = signals.shape[0]
            mask = torch.zeros(batch_size, self.num_patches, device=signals.device)
            ids_restore = torch.arange(
                self.num_patches,
                device=signals.device,
            ).unsqueeze(0).repeat(batch_size, 1)

        cls_tokens = self.cls_token + self.pos_embed[:, :1]
        cls_tokens = cls_tokens.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens, mask, ids_restore

    def forward_decoder(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.decoder_embed(latent)
        tokens_without_cls = tokens[:, 1:, :]

        mask_tokens = self.mask_token.repeat(
            tokens.shape[0],
            ids_restore.shape[1] + 1 - tokens_without_cls.shape[1],
            1,
        )
        tokens_ = torch.cat((tokens_without_cls, mask_tokens), dim=1)
        tokens_ = torch.gather(
            tokens_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, tokens.shape[2]),
        )
        tokens = torch.cat((tokens[:, :1, :], tokens_), dim=1)
        tokens = tokens + self.decoder_pos_embed[:, : tokens.shape[1]]

        for block in self.decoder_blocks:
            tokens = block(tokens)
        tokens = self.decoder_norm(tokens)
        tokens = self.decoder_pred(tokens)
        return tokens[:, 1:, :]

    def forward_loss(
        self,
        signals: torch.Tensor,
        pred_patches: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self.patchify(signals)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            variance = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (variance + 1.0e-6).sqrt()

        loss = (pred_patches - target).pow(2).mean(dim=-1)
        if mask.sum() == 0:
            return loss.mean()
        return (loss * mask).sum() / mask.sum()

    def encode(self, signals: torch.Tensor) -> torch.Tensor:
        latent, _, _ = self.forward_encoder(signals, mask_ratio=0.0)
        return latent[:, 1:, :]

    def decode(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.decoder_embed(latent_tokens)
        tokens = tokens + self.decoder_pos_embed[:, 1 : latent_tokens.shape[1] + 1]
        for block in self.decoder_blocks:
            tokens = block(tokens)
        tokens = self.decoder_norm(tokens)
        pred = self.decoder_pred(tokens)
        return self.unpatchify(pred)

    def forward(self, signals: torch.Tensor, mask_ratio: float = 0.75) -> EEGMAEOutput:
        latent, mask, ids_restore = self.forward_encoder(signals, mask_ratio=mask_ratio)
        pred_patches = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(signals, pred_patches, mask)
        reconstruction = self.unpatchify(pred_patches)
        return EEGMAEOutput(
            loss=loss,
            reconstruction=reconstruction,
            latent=latent[:, 1:, :],
            mask=mask,
        )
