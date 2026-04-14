from __future__ import annotations

import math

import torch
from torch import nn


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    batch_size = timesteps.shape[0]
    out = values.gather(-1, timesteps.to(values.device)).to(timesteps.device)
    return out.reshape(batch_size, *([1] * (len(target_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1.0e-5, 0.999).float()


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(1.0e-4, 0.02, timesteps, dtype=torch.float32)


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "cosine") -> None:
        super().__init__()
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.ones(1), alphas_cumprod[:-1]), dim=0)

        self.timesteps = timesteps
        self.register_buffer("betas", betas, persistent=True)
        self.register_buffer("alphas", alphas, persistent=True)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=True)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev, persistent=True)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=True)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=True,
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod),
            persistent=True,
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
            persistent=True,
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance, persistent=True)

    def sample_timesteps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device)

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = noise if noise is not None else torch.randn_like(x_start)
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def p_sample(
        self,
        denoiser: nn.Module,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        pred_noise = denoiser(x_t, timesteps, condition_tokens)
        x0_pred = self.predict_start_from_noise(x_t, timesteps, pred_noise).clamp(-5.0, 5.0)

        beta_t = _extract(self.betas, timesteps, x_t.shape)
        alpha_t = _extract(self.alphas, timesteps, x_t.shape)
        alpha_cumprod_t = _extract(self.alphas_cumprod, timesteps, x_t.shape)
        alpha_cumprod_prev_t = _extract(self.alphas_cumprod_prev, timesteps, x_t.shape)

        mean = (
            beta_t * torch.sqrt(alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t) * x0_pred
            + (1.0 - alpha_cumprod_prev_t) * torch.sqrt(alpha_t) / (1.0 - alpha_cumprod_t) * x_t
        )
        variance = _extract(self.posterior_variance, timesteps, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = (timesteps != 0).float().reshape(x_t.shape[0], *([1] * (x_t.ndim - 1)))
        return mean + nonzero_mask * variance.sqrt() * noise

    @torch.no_grad()
    def sample_loop(
        self,
        denoiser: nn.Module,
        shape: tuple[int, ...],
        condition_tokens: torch.Tensor,
        device: torch.device | str,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        sample = torch.randn(shape, device=device)
        step_indices = list(range(self.timesteps - 1, -1, -1))
        if num_steps is not None and num_steps < self.timesteps:
            stride = max(1, self.timesteps // num_steps)
            step_indices = list(range(self.timesteps - 1, -1, -stride))
            if step_indices[-1] != 0:
                step_indices.append(0)

        for step in step_indices:
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            sample = self.p_sample(denoiser, sample, timesteps, condition_tokens)
        return sample
