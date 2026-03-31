"""
Noise schedules for discrete (absorbing-state) diffusion.
Ported from mdlm/noise_schedule.py.
"""

import abc
import torch
import torch.nn as nn


class Noise(abc.ABC, nn.Module):
    """Base noise schedule: returns (total_noise, rate_noise)."""

    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """g(t) = d sigma / dt"""

    @abc.abstractmethod
    def total_noise(self, t):
        """sigma(t) = integral_0^t g(s) ds"""


class LogLinearNoise(Noise):
    """Log-linear noise schedule (MDLM default).

    sigma(t) = -log(1 - (1-eps)*t)
    so that  1 - exp(-sigma(t)) = (1-eps)*t  linearly interpolates
    the *mask probability* from 0 to ~1.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def importance_sampling_transformation(self, t):
        f_T = torch.log1p(-torch.exp(-self.sigma_max))
        f_0 = torch.log1p(-torch.exp(-self.sigma_min))
        sigma_t = -torch.log1p(-torch.exp(t * f_T + (1 - t) * f_0))
        t = -torch.expm1(-sigma_t) / (1 - self.eps)
        return t


class CosineNoise(Noise):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(self, t):
        cos = (1 - self.eps) * torch.cos(t * torch.pi / 2)
        sin = (1 - self.eps) * torch.sin(t * torch.pi / 2)
        scale = torch.pi / 2
        return scale * sin / (cos + self.eps)

    def total_noise(self, t):
        cos = torch.cos(t * torch.pi / 2)
        return -torch.log(self.eps + (1 - self.eps) * cos)


def get_noise(noise_type: str = "loglinear", eps: float = 1e-3) -> Noise:
    if noise_type == "loglinear":
        return LogLinearNoise(eps=eps)
    elif noise_type == "cosine":
        return CosineNoise(eps=eps)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
