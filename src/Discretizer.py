import torch
import torch.nn as nn
import torch.nn.functional as F

class FSQDiscretizer(nn.Module):
    def __init__(self, slot_dim: int, levels: list[int], fsq_dim: int = None,
                 drop_quant_p: float = 0.0, corrupt_tokens_p: float = 0.0):
        super().__init__()
        self.fsq = FSQ(
            latents_read_key="z",
            quants_write_key="q",
            tokens_write_key="t",
            levels=levels,
            drop_quant_p=drop_quant_p,
            corrupt_tokens_p=corrupt_tokens_p,
        )
        self.fsq_dim = self.fsq.dim
        assert fsq_dim is None or fsq_dim == self.fsq_dim

        # slot_dim -> fsq_dim -> slot_dim
        self.pre  = nn.Linear(slot_dim, self.fsq_dim)
        self.post = nn.Linear(self.fsq_dim, slot_dim)

    def encode(self, slots: torch.Tensor):
        """
        slots: (B,K,slot_dim)
        returns:
          quant_slots: (B,K,slot_dim)  (diffusion conditioning에 바로 넣을 수 있게 post까지)
          tokens: (B,K) long
        """
        z = self.pre(slots)                  # (B,K,fsq_dim)
        q, t = self.fsq.forward_z(z)         # q: (B,K,fsq_dim), t: (B,K)
        quant_slots = self.post(q)           # (B,K,slot_dim)
        return quant_slots, t

    def decode(self, tokens: torch.Tensor):
        """
        tokens: (B,K) long
        returns:
          quant_slots: (B,K,slot_dim)
        """
        # FSQ indices_to_embedding: (...)->(...,fsq_dim) in [-1,1]
        q = self.fsq.indices_to_embedding(tokens)   # (B,K,fsq_dim)
        quant_slots = self.post(q)
        return quant_slots

    def forward(self, slots: torch.Tensor):
        return self.encode(slots)

class VQDiscretizer(nn.Module):
    """
    VQ-VAE style vector quantizer wrapper.
    Interface-compatible with your FSQDiscretizer:
      forward(slots) -> (quant_slots, tokens)

    slots: (B,K,slot_dim)
    returns:
      quant_slots: (B,K,slot_dim)
      tokens: (B,K) long in [0, codebook_size-1]
    """
    def __init__(
        self,
        slot_dim: int,
        codebook_size: int = 512,
        vq_dim: int | None = None,          # 내부 quantize 차원 (None이면 slot_dim)
        beta: float = 0.25,                 # commitment weight (non-EMA에서 사용)
        use_ema: bool = False,               # EMA codebook update 추천
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        drop_quant_p: float = 0.0,          # training 시 일정 확률로 quant bypass(soft regularization)
        corrupt_tokens_p: float = 0.0,      # training 시 일정 확률로 토큰 랜덤 교체
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.codebook_size = int(codebook_size)
        self.vq_dim = int(vq_dim) if vq_dim is not None else int(slot_dim)

        self.beta = float(beta)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)

        self.drop_quant_p = float(drop_quant_p)
        self.corrupt_tokens_p = float(corrupt_tokens_p)

        # slot_dim -> vq_dim -> slot_dim (FSQDiscretizer와 동일한 패턴)
        self.pre = nn.Sequential(
            nn.Linear(slot_dim, self.vq_dim),
            nn.LayerNorm(self.vq_dim),
        )
        # self.pre = nn.Linear(self.slot_dim, self.vq_dim)
        self.post = nn.Linear(self.vq_dim, self.slot_dim)

        # codebook
        self.codebook = nn.Embedding(self.codebook_size, self.vq_dim)
        nn.init.uniform_(self.codebook.weight, -1.0, 1.0)
        # nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

        # EMA state
        if self.use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
            self.register_buffer("ema_embed_avg", torch.zeros(self.codebook_size, self.vq_dim))
            self.codebook.weight.requires_grad_(False)

        # 디버깅/로그용(원하면 밖에서 읽어)
        self.last_vq_loss = None
        self.last_perplexity = None
        self.last_usage = None

    @torch.no_grad()
    def _ema_update(self, encodings: torch.Tensor, flat_z: torch.Tensor):
        """
        encodings: (N, codebook_size) one-hot
        flat_z:    (N, vq_dim)
        """
        # cluster size
        cluster_size = encodings.sum(dim=0)  # (M,)
        # embedding sum
        embed_sum = encodings.t() @ flat_z   # (M, D)

        # EMA
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size * (1.0 - self.ema_decay))
        self.ema_embed_avg.mul_(self.ema_decay).add_(embed_sum * (1.0 - self.ema_decay))

        # laplace smoothing
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.ema_eps) / (n + self.codebook_size * self.ema_eps) * n

        # normalize to get new embeddings
        new_embed = self.ema_embed_avg / smoothed.unsqueeze(1).clamp_min(1e-12)
        self.codebook.weight.data.copy_(new_embed)

    def encode(self, slots: torch.Tensor):
        """
        slots: (B,K,slot_dim)
        returns:
          quant_slots: (B,K,slot_dim)
          tokens: (B,K) long
        """
        B, K, _ = slots.shape

        z = self.pre(slots)                 # (B,K,vq_dim)
        flat_z = z.reshape(-1, self.vq_dim) # (N,vq_dim), N=B*K

        # --- NN search in codebook ---
        # dist(x,e) = ||x||^2 + ||e||^2 - 2 x·e
        cb = F.layer_norm(self.codebook.weight, (self.vq_dim,))           # (M,vq_dim)
        # cb = self.codebook.weight
        x2 = (flat_z ** 2).sum(dim=1, keepdim=True)         # (N,1)
        e2 = (cb ** 2).sum(dim=1).unsqueeze(0)              # (1,M)
        xe = flat_z @ cb.t()                                 # (N,M)
        dist = x2 + e2 - 2.0 * xe                             # (N,M)

        token_ids = torch.argmin(dist, dim=1)                 # (N,)
        token_ids = token_ids.view(B, K)                      # (B,K)

        # (선택) token corruption
        if self.training and self.corrupt_tokens_p > 0.0:
            mask = (torch.rand(B, K, device=token_ids.device) < self.corrupt_tokens_p)
            rand_ids = torch.randint(0, self.codebook_size, (B, K), device=token_ids.device)
            token_ids = torch.where(mask, rand_ids, token_ids)

        # quant vectors
        quant = self.codebook(token_ids)                      # (B,K,vq_dim)

        # (선택) quant drop (bypass)
        if self.training and self.drop_quant_p > 0.0:
            drop = (torch.rand(B, K, 1, device=quant.device) < self.drop_quant_p)
            quant = torch.where(drop, z, quant)               # quant 대신 원래 z 사용

        # straight-through estimator
        quant_st = z + (quant - z).detach()                   # (B,K,vq_dim)

        # post proj back to slot_dim
        quant_slots = self.post(quant_st)                     # (B,K,slot_dim)

        # ---------- losses & stats ----------
        # VQ-VAE loss (non-EMA only). EMA update uses only commitment loss.
        if self.training:
            if self.use_ema:
                # EMA update uses one-hot assignments
                with torch.no_grad():
                    flat_ids = token_ids.reshape(-1)          # (N,)
                    encodings = F.one_hot(flat_ids, self.codebook_size).type_as(flat_z)  # (N,M)
                    self._ema_update(encodings, flat_z)

                # commitment loss encourages encoder outputs to stay near chosen embedding
                commit = F.mse_loss(z, quant.detach())
                vq_loss = self.beta * commit
            else:
                # classic VQ-VAE
                codebook_loss = F.mse_loss(quant, z.detach())
                commit_loss = F.mse_loss(z, quant.detach())
                vq_loss = codebook_loss + self.beta * commit_loss

            self.last_vq_loss = vq_loss

            # perplexity / usage
            with torch.no_grad():
                flat_ids = token_ids.reshape(-1)
                hist = torch.bincount(flat_ids, minlength=self.codebook_size).float()
                prob = hist / hist.sum().clamp_min(1.0)
                perplexity = torch.exp(-(prob[prob > 0] * torch.log(prob[prob > 0])).sum())
                usage = (hist > 0).float().mean()
                self.last_perplexity = perplexity
                self.last_usage = usage
        else:
            self.last_vq_loss = None

        return quant_slots, token_ids

    def decode(self, tokens: torch.Tensor):
        """
        tokens: (B,K) long
        returns:
          quant_slots: (B,K,slot_dim)
        """
        q = self.codebook(tokens)   # (B,K,vq_dim)
        return self.post(q)

    def forward(self, slots: torch.Tensor):
        return self.encode(slots)



class Discretizer(nn.Module):
    def __init__(self, codebook_size: int, slot_dim: int, temperature: float = 1.0):
        """
        Args:
            codebook_size: Number of discrete codes (V)
            slot_dim: Dimensionality of each slot/code (D)
            temperature: Temperature for Gumbel-Softmax
            hard: Whether to sample hard one-hot vectors (straight-through)
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.slot_dim = slot_dim
        self.temperature = temperature

        self.codebook = nn.Parameter(torch.randn(codebook_size, slot_dim))
        nn.init.normal_(self.codebook, std=0.02)

    def forward(self, slots):
        """
        Args:
            slots: Tensor of shape (B, K, D) - continuous slots

        Returns:
            discretized: Tensor of shape (B, K, D)
            code_indices: Tensor of shape (B, K) with indices of selected codes
        """
        B, K, D = slots.shape
        assert D == self.slot_dim, f"Slot dim mismatch: {D} != {self.slot_dim}"

        # (B*K, D)
        slots_flat = slots.view(-1, D)
        codebook_norm = F.normalize(self.codebook, dim=-1)  # (V, D)
        slots_norm = F.normalize(slots_flat, dim=-1)        # (B*K, D)

        # Cosine similarity -> (B*K, V)
        sim = torch.matmul(slots_norm, codebook_norm.T)

        # Gumbel softmax
        probs = gumbel_softmax(sim, temperature=self.temperature)  # (B*K, V)

        # Weighted sum over codebook vectors -> (B*K, D)
        quantized = torch.matmul(probs, self.codebook)  # (B*K, D)
        quantized = quantized.view(B, K, D)

        code_indices = probs.argmax(dim=-1).view(B, K)

        return quantized, code_indices



def gumbel_softmax(logits, temperature, device=None):
    """
    Taken from https://github.com/dev4488/VAE_gumble_softmax/blob/master/vae_gumbel_softmax.py

    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        device = logits.device

    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape).to(device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(logits, temperature):
        y = logits + sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    # (bs, c, out_dim//c)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard

# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.
# --------------------------------------------------------
# Adapted from:
# https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/finite_scalar_quantization.py
# Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
# Code adapted from Jax version in Appendix A.1
# --------------------------------------------------------

import random
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from einops import pack, rearrange, repeat, unpack

import torch
import torch.nn as nn
from torch import Tensor, int32
from torch.amp import autocast
from torch.nn import Module


def round_ste_quant_dropout(z: Tensor, drop_quant_p: float) -> Tensor:
    """Round with straight through gradients, randomly skip quantization per sample."""
    zhat = z.round()
    batch_size = z.shape[0]
    device = z.device
    # Create a mask where each sample has a probability `drop_quant_p` to not be quantized
    mask = torch.bernoulli(torch.full((batch_size,), drop_quant_p, device=device))
    # Reshape mask to broadcast over the remaining dimensions
    mask = mask.view(batch_size, *([1] * (z.ndim - 1)))
    # Apply the mask: if mask=1, keep the original value; if mask=0, use the quantized value
    output = z + ((1 - mask) * (zhat - z)).detach()
    return output


class FSQ(nn.Module):
    """Minimal FSQ (https://arxiv.org/abs/2309.15505) implementation. Except when using
    packed_call, expects channel dimension to be last.

    Args:
        latents_read_key: Dictionary key to read input latents from.
        quants_write_key: Dictionary key to write quantized latents.
        tokens_write_key: Dictionary key to write discrete token ids.
        levels: List of FSQ levels. See https://arxiv.org/abs/2309.15505 for suggestions.
        drop_quant_p: During training, pass the non-rounded values with this probability for each
            sample in the batch/list.
        corrupt_tokens_p: During training, optionally corrupt tokens by setting a percentage of them
            to random other tokens indices.
        min_corrupt_tokens_p: Optional argument specifying a minimum percentage of tokens to be
            corrupted. The actual percentage is sampled uniformly between the min and max per sample.
        apply_corrupt_tokens_p: Probability of activating token corruption per sample. Only active if
            corrupt_tokens_p > 0.
        packed_call: Set to True to pack list of examples and quantize them jointly. Might be slighly
            more efficient.
    """

    def __init__(
        self,
        latents_read_key: str,
        quants_write_key: str,
        tokens_write_key: str,
        levels: List[int],
        drop_quant_p: float = 0.0,
        corrupt_tokens_p: float = 0.0,
        min_corrupt_tokens_p: Optional[float] = None,
        apply_corrupt_tokens_p: float = 0.2,
        packed_call: bool = True,
    ):
        super().__init__()
        self.latents_read_key = latents_read_key
        self.quants_write_key = quants_write_key
        self.tokens_write_key = tokens_write_key

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.dim = len(levels)
        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_embedding(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.drop_quant_p = drop_quant_p
        self.corrupt_tokens_p = corrupt_tokens_p
        self.min_corrupt_tokens_p = min_corrupt_tokens_p or corrupt_tokens_p
        self.apply_corrupt_tokens_p = apply_corrupt_tokens_p

    def __repr__(self):
        return f"FSQ(levels={self._levels.tolist()}, codebook_size={self.codebook_size})"
    
    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        bounded = self.bound(z)
        drop_quant_p = self.drop_quant_p if self.training else 0.0
        quantized = round_ste_quant_dropout(bounded, drop_quant_p)
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_embedding(
        self,
        indices: Tensor,
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        return codes

    def corrupt_quant(self, quant: Tensor) -> Tensor:
        "Randomly corrupt some entries of the quantized Tensor"
        quant_shape, quant_device = quant.shape[:-1], quant.device
        random_indices = torch.randint(
            low=0, high=self.codebook_size, size=quant_shape, device=quant_device
        )
        random_quant = self.implicit_codebook[random_indices]
        sample_corrupt_tokens_p = random.uniform(self.min_corrupt_tokens_p, self.corrupt_tokens_p)
        corruption_mask = torch.rand(quant_shape, device=quant_device) < sample_corrupt_tokens_p
        corruption_mask = repeat(corruption_mask, "... -> ... d", d=quant.shape[-1])
        return torch.where(corruption_mask, random_quant, quant)

    @autocast(device_type="cuda", enabled=False)
    def forward_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        quant = self.quantize(z.float())
        if (
            self.training
            and self.corrupt_tokens_p > 0.0
            and random.random() < self.apply_corrupt_tokens_p
        ):
            # Optionally corrupt a random percentage of tokens during training
            quant = self.corrupt_quant(quant)
        tokens = self.codes_to_indices(quant)

        # Incompatible return value type (got "Tensor", expected "LongTensor")  [return-value]
        return quant, tokens.long()  # type: ignore

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        z = data_dict[self.latents_read_key]

        if isinstance(z, list):
            quant, tokens = [], []
            for z_i in z:
                quant_i, tokens_i = self.forward_z(z_i)
                quant.append(quant_i)
                tokens.append(tokens_i)
        else:
            quant, tokens = self.forward_z(z)

        data_dict[self.quants_write_key] = quant
        data_dict[self.tokens_write_key] = tokens

        return data_dict