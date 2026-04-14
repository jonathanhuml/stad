import torch


def masked_reconstruction_loss(loss: torch.Tensor) -> torch.Tensor:
    return loss


def latent_diffusion_loss(pred_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred_noise, target_noise)

