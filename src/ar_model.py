"""
Autoregressive (AR) model for discrete token generation.

Same interface as DiscreteDiffusion (compute_loss / sample) so they
can be swapped in train_discrete_diffusion_v2.py with minimal changes.

The backbone is the same DIT with causal=True.  Training uses standard
next-token-prediction cross-entropy.  Sampling uses left-to-right
generation with temperature / top-k / top-p.

Conditioning strategy:
  - Class labels (ImageNet): adaLN — same as diffusion, natural for AR.
  - Structured conditions (CLEVR etc.): **prefix concat** — condition
    tokens are prepended to the data sequence so the AR model sees them
    as context, just like a prompt.  No cross-attention needed.
  - No CFG — AR models don't train with mask-based noising, so the
    cond/uncond split doesn't apply cleanly.  Use temperature/top-p
    for diversity control instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dit_model import DIT


@dataclass
class LossOutput:
    loss: Tensor        # scalar
    nlls: Tensor        # (B, L) per-token NLL
    token_mask: Tensor  # (B, L)


def _top_k_top_p_filter(logits: Tensor, top_k: int = 0, top_p: float = 1.0):
    """Filter logits with top-k and/or nucleus (top-p) sampling."""
    if top_k > 0:
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits = logits.scatter(
            -1, sorted_idx,
            sorted_logits.masked_fill(remove, -float("inf")))
    return logits


class AutoregressiveModel(nn.Module):
    """
    Autoregressive model wrapping a causal DIT backbone.

    Conditioning is done via **prefix concat**: external condition
    embeddings (e.g. from CLEVRConditionEncoder) are prepended to the
    data token sequence.  The causal backbone sees [cond_prefix | data]
    and we only compute loss / generate on the data portion.

    Parameters
    ----------
    backbone : DIT
        A DIT with causal=True.
    vocab_size : int
        Number of *data* token categories.
    cond_proj : nn.Module or None
        Projects external condition tokens to backbone hidden_size.
        If None, cond_tokens must already match hidden_size.
    """

    def __init__(self, backbone: nn.Module, vocab_size: int,
                 cond_proj: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        self.data_vocab_size = vocab_size
        # mask_index kept for eval pipeline compat (e.g. visualization)
        self.mask_index = vocab_size
        self.cond_proj = cond_proj

    def _get_hidden_size(self) -> int:
        return self.backbone.token_emb.embedding_dim

    def _embed_and_concat(
        self,
        x: Tensor,
        cond_tokens: Optional[Tensor],
    ) -> tuple[Tensor, int]:
        """Embed data tokens and prepend condition prefix.

        Args:
            x:           (B, L) int64 data tokens
            cond_tokens: (B, L_c, D) float condition embeddings, or None
        Returns:
            h:           (B, L_c + L, D) concatenated hidden states
            n_prefix:    number of prefix positions (L_c or 0)
        """
        B, L = x.shape
        D = self._get_hidden_size()

        # embed data tokens  (re-use backbone's embedding + pos_emb)
        h_data = self.backbone.token_emb(x.long())          # (B, L, D)
        h_data = self.backbone.pos_emb(h_data)

        # add structured positional embeddings (2d / sudoku / multires)
        if self.backbone.pos_emb_type in ("2d", "sudoku"):
            pos = (self.backbone.row_emb(self.backbone.row_idx) +
                   self.backbone.col_emb(self.backbone.col_idx))
            if self.backbone.pos_emb_type == "sudoku":
                pos = pos + self.backbone.box_emb(self.backbone.box_idx)
            h_data = h_data + pos[None, :, :]
        elif self.backbone.pos_emb_type == "multires":
            pos = (self.backbone.mr_level_emb(self.backbone.mr_level_idx) +
                   self.backbone.mr_row_emb(self.backbone.mr_row_idx) +
                   self.backbone.mr_col_emb(self.backbone.mr_col_idx))
            h_data = h_data + pos[None, :, :]

        if cond_tokens is None:
            return h_data, 0

        # project condition if needed
        if self.cond_proj is not None:
            cond_tokens = self.cond_proj(cond_tokens)

        n_prefix = cond_tokens.shape[1]
        h = torch.cat([cond_tokens, h_data], dim=1)  # (B, L_c + L, D)
        return h, n_prefix

    def _forward_hidden(
        self,
        h: Tensor,
        sigma: Tensor,
        class_labels: Optional[Tensor],
    ) -> Tensor:
        """Run transformer blocks + output layer on hidden states.

        Args:
            h:            (B, L_total, D) hidden states
            sigma:        (B,) dummy timestep (zeros for AR)
            class_labels: (B,) optional class labels for adaLN
        Returns:
            logits: (B, L_total, vocab_size)
        """
        c = F.silu(self.backbone.sigma_map(sigma))
        if class_labels is not None and self.backbone.label_emb is not None:
            c = c + F.silu(self.backbone.label_emb(class_labels))

        for blk in self.backbone.blocks:
            # pass cond_tokens=None to blocks — conditioning is already
            # in the prefix via concat, no cross-attention needed
            h = blk(h, c, cond_tokens=None)

        h = self.backbone.output_layer(h, c)  # (B, L_total, V)
        return h

    # ─────────────── training loss ─────────────────────────

    def compute_loss(
        self,
        x0: Tensor,
        attention_mask: Optional[Tensor] = None,
        cond_tokens: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
    ) -> LossOutput:
        """Next-token prediction cross-entropy.

        x0: (B, L) int64 clean token sequence.

        If cond_tokens is provided, they are prepended as a prefix:
            [cond_0, ..., cond_{C-1}, x0_0, ..., x0_{L-1}]
        Loss is only computed on the data portion (shifted):
            logits[C-1 .. C+L-2] predict x0[0 .. L-1]
        So we get L predictions total (including predicting x0[0] from
        the last cond token).
        """
        B, L = x0.shape

        h, n_prefix = self._embed_and_concat(x0, cond_tokens)
        sigma = torch.zeros(B, device=x0.device)
        logits = self._forward_hidden(h, sigma, class_labels)  # (B, C+L, V)

        # slice to data portion: we want logits that predict x0[0..L-1]
        # In the full sequence [cond | data], position C+i has seen
        # everything up to C+i. Logits at position C+i-1 predict x0[i].
        # So: logits[:, C-1 : C+L-1] predict x0[0 : L].
        # But we can only predict up to x0[L-1], so:
        #   pred_logits = logits[:, (C-1 or 0) : C+L-1]
        #   targets     = x0[:, 0 : L] (or x0[:, 1:L] if no prefix)
        if n_prefix > 0:
            # logits[C-1] predicts x0[0], logits[C] predicts x0[1], ...
            pred_logits = logits[:, n_prefix - 1: n_prefix + L - 1,
                                :self.data_vocab_size]   # (B, L, V)
            targets = x0                                  # (B, L)
            if attention_mask is None:
                attention_mask = torch.ones(B, L, dtype=torch.float32,
                                            device=x0.device)
        else:
            # No prefix: standard shifted loss
            # logits[i] predicts x0[i+1]
            pred_logits = logits[:, :L - 1,
                                :self.data_vocab_size]    # (B, L-1, V)
            targets = x0[:, 1:]                           # (B, L-1)
            if attention_mask is None:
                attention_mask = torch.ones(B, L - 1, dtype=torch.float32,
                                            device=x0.device)
            else:
                attention_mask = attention_mask[:, 1:]

        log_probs = F.log_softmax(pred_logits, dim=-1)
        nll = -torch.gather(log_probs, dim=-1,
                            index=targets[:, :, None]).squeeze(-1)

        masked_nll = nll * attention_mask
        loss = masked_nll.sum() / attention_mask.sum().clamp(min=1)

        return LossOutput(loss=loss, nlls=masked_nll, token_mask=attention_mask)

    # ─────────────── sampling ─────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device = torch.device("cpu"),
        cond_tokens: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        return_history: bool = False,
        # accepted but ignored (interface compat with diffusion)
        num_steps: int = 0,
        sampler: str = "ar",
        noise_removal: bool = True,
        tokens_per_step: int = 0,
        guidance_scale: float = 1.0,  # ignored for AR
    ) -> Tensor:
        """Left-to-right autoregressive sampling.

        If cond_tokens is provided, they form a prefix that the model
        attends to as context when generating each data token.

        Args:
            batch_size:  number of samples
            seq_len:     number of *data* tokens to generate
            temperature: softmax temperature
            top_k:       top-k filtering (0 = disabled)
            top_p:       nucleus sampling (1.0 = disabled)
        Returns:
            x: (batch_size, seq_len) int64 generated tokens
        """
        x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        sigma = torch.zeros(batch_size, device=device)
        history = [] if return_history else None

        for i in range(seq_len):
            h, n_prefix = self._embed_and_concat(x, cond_tokens)
            logits = self._forward_hidden(h, sigma, class_labels)

            # Which position's logits predict x[i]?
            if n_prefix > 0:
                # logits[n_prefix - 1 + i] predicts x[i]
                pred_pos = n_prefix - 1 + i
            else:
                # No prefix: logits[i-1] predicts x[i], except i==0
                pred_pos = max(0, i - 1) if i > 0 else 0

            next_logits = logits[:, pred_pos, :self.data_vocab_size]

            # temperature + filtering
            next_logits = next_logits / max(temperature, 1e-8)
            next_logits = _top_k_top_p_filter(next_logits, top_k=top_k,
                                               top_p=top_p)

            probs = F.softmax(next_logits, dim=-1)
            x[:, i] = torch.multinomial(probs, num_samples=1).squeeze(-1)

            if return_history:
                history.append(x.clone().cpu())

        if return_history:
            return x, history
        return x

    @torch.no_grad()
    def sample_inpaint(
        self,
        x_gt: Tensor,
        known_mask: Tensor,
        cond_tokens: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        return_history: bool = False,
        # ignored for compat
        num_steps: int = 0,
        sampler: str = "ar",
        noise_removal: bool = True,
        tokens_per_step: int = 0,
        guidance_scale: float = 1.0,
        return_confidence_history: bool = False,
        return_step_logs: bool = False,
    ):
        """AR inpainting: known positions use GT, unknown are sampled."""
        device = x_gt.device
        B, L = x_gt.shape
        sigma = torch.zeros(B, device=device)

        x = torch.where(known_mask, x_gt, torch.zeros_like(x_gt))
        history = [torch.where(known_mask, x_gt,
                               torch.full_like(x_gt, self.mask_index)).cpu()] \
            if return_history else None

        for i in range(L):
            if known_mask[:, i].all():
                continue

            h, n_prefix = self._embed_and_concat(x, cond_tokens)
            logits = self._forward_hidden(h, sigma, class_labels)

            pred_pos = (n_prefix - 1 + i) if n_prefix > 0 else max(0, i - 1)
            next_logits = logits[:, pred_pos, :self.data_vocab_size]

            next_logits = next_logits / max(temperature, 1e-8)
            next_logits = _top_k_top_p_filter(next_logits, top_k=top_k,
                                               top_p=top_p)
            probs = F.softmax(next_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

            x[:, i] = torch.where(known_mask[:, i], x_gt[:, i], sampled)

            if return_history:
                full = x.clone()
                for j in range(i + 1, L):
                    full[:, j] = torch.where(
                        known_mask[:, j], x_gt[:, j],
                        torch.full((B,), self.mask_index, device=device))
                history.append(full.cpu())

        if return_history:
            return x, history
        return x
