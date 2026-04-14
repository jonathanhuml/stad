from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.diffusion import GaussianDiffusion
from models.eeg_mae import EEGMAE
from models.mtd import MTD
from models.stc import STC


class STAD(nn.Module):
    def __init__(
        self,
        mae: EEGMAE,
        stc: STC,
        mtd: MTD,
        diffusion: GaussianDiffusion,
        freeze_mae: bool = True,
    ) -> None:
        super().__init__()
        self.mae = mae
        self.stc = stc
        self.mtd = mtd
        self.diffusion = diffusion
        self.freeze_mae = freeze_mae
        if freeze_mae:
            for parameter in self.mae.parameters():
                parameter.requires_grad = False

    def encode_hr(self, hr_eeg: torch.Tensor) -> torch.Tensor:
        if self.freeze_mae:
            with torch.no_grad():
                return self.mae.encode(hr_eeg)
        return self.mae.encode(hr_eeg)

    def decode_latent(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        if self.freeze_mae:
            with torch.no_grad():
                return self.mae.decode(latent_tokens)
        return self.mae.decode(latent_tokens)

    def forward(
        self,
        lr_eeg: torch.Tensor,
        hr_eeg: torch.Tensor,
        lr_positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode_hr(hr_eeg)
        condition_tokens = self.stc(lr_eeg, lr_positions)
        timesteps = self.diffusion.sample_timesteps(hr_eeg.shape[0], hr_eeg.device)
        noise = torch.randn_like(latent)
        noisy_latent = self.diffusion.q_sample(latent, timesteps, noise)
        pred_noise = self.mtd(noisy_latent, timesteps, condition_tokens)
        loss = F.mse_loss(pred_noise, noise)
        return {
            "loss": loss,
            "latent": latent,
            "condition_tokens": condition_tokens,
            "pred_noise": pred_noise,
            "noise": noise,
        }

    @torch.no_grad()
    def reconstruct(
        self,
        lr_eeg: torch.Tensor,
        lr_positions: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        condition_tokens = self.stc(lr_eeg, lr_positions)
        latent = self.diffusion.sample_loop(
            denoiser=self.mtd,
            shape=(lr_eeg.shape[0], self.mae.num_patches, self.mae.embed_dim),
            condition_tokens=condition_tokens,
            device=lr_eeg.device,
            num_steps=num_steps,
        )
        return self.decode_latent(latent)
