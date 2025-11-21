# sampling.py

from typing import Optional

import torch
from torch import nn
from diffusers import DDIMScheduler


@torch.no_grad()
def sample_ddim_with_cfg(
    model: nn.Module,
    noise_scheduler: DDIMScheduler,
    cond: torch.Tensor,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    uncond_cond: Optional[torch.Tensor] = None,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    model.eval()

    cond = cond.to(device)
    batch_size = cond.shape[0]

    unet = model.unet
    in_channels = unet.config.in_channels
    image_size = unet.sample_size

    latents = torch.randn(
        (batch_size, in_channels, image_size, image_size),
        device=device,
        generator=generator,
    )

    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    use_cfg = (uncond_cond is not None) and (guidance_scale is not None) and (guidance_scale != 1.0)
    if use_cfg:
        uncond_cond = uncond_cond.to(device)

    for t in noise_scheduler.timesteps:
        t_batch = torch.full(
            (batch_size,),
            t,
            device=device,
            dtype=torch.long,
        )

        if use_cfg:
            # 1) uncond: y = uncond_cond
            noise_pred_uncond = model(latents, t_batch, y=uncond_cond)

            # 2) cond: y = cond
            noise_pred_cond = model(latents, t_batch, y=cond)

            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred = model(latents, t_batch, y=cond)

        step_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=latents,
            eta=eta,
            generator=generator,
        )
        latents = step_output.prev_sample

    return latents.clamp(-1.0, 1.0)
