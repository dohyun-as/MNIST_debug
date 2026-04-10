"""
Text-Conditioned DiT (Baseline)
================================

Baseline for comparison: encode CLEVR structured conditions using a
**pretrained language model** (T5, CLIP), then condition a DiT via
cross-attention — mirroring how real text-to-image diffusion works
(Imagen, PixArt-α, Stable Diffusion).

Pipeline:
  CLEVR JSON → natural language text → pretrained LM (frozen) →
  learned projection → cross-attention in DiT → image

Also supports a from-scratch CLEVRTextEncoder for ablation.

Architecture reuses DiT blocks from model_multires.py.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_multires import (
    _RMSNorm,
    _SwiGLUFFN,
    _DiTBlock,
    _DiTFinalLayer,
    _BottleneckPatchEmbed,
    _sinusoidal_timestep_embedding,
    _get_2d_sincos_pos_embed,
    _build_2d_rope,
    _rotate_half,
)


# ────────────────────────────────────────────────────────────
#  CLEVR JSON → natural language text
# ────────────────────────────────────────────────────────────

def clevr_json_to_text(cond_json: dict) -> str:
    """Convert CLEVR condition JSON to natural language description.

    Example output:
      "Scene with 3 objects. Object A: small red rubber cube.
       Object B: large blue metal sphere. Object C: small green rubber cylinder.
       A is left of B. C is behind A."
    """
    entities = cond_json.get("entities", [])
    relations = cond_json.get("relations", [])

    parts = [f"Scene with {len(entities)} objects."]

    for ent in entities:
        attrs = ent["attrs"]
        name = ent["name"]
        parts.append(
            f"Object {name}: {attrs['size']} {attrs['color']} "
            f"{attrs['material']} {attrs['shape']}."
        )

    for rel in relations:
        rel_text = rel["rel"].replace("_", " ")
        parts.append(f"{rel['subj']} is {rel_text} {rel['obj']}.")

    return " ".join(parts)


# ────────────────────────────────────────────────────────────
#  Pretrained LM text encoder wrapper
# ────────────────────────────────────────────────────────────

class PretrainedTextEncoder(nn.Module):
    """Wraps a pretrained language model for text conditioning.

    Supports:
      - T5 (encoder-only): google/t5-small, google/t5-base, etc.
        Used by PixArt-α, Imagen. No token limit practically.
      - CLIP text encoder: openai/clip-vit-base-patch32, etc.
        Used by Stable Diffusion 1.x. 77 token limit.

    The LM is frozen; only the learned projection layer is trained.
    """

    def __init__(self, model_name: str = "google-t5/t5-base",
                 output_dim: int = 768, max_length: int = 256,
                 freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.freeze = freeze
        self._model_type = None

        if "t5" in model_name.lower():
            self._init_t5(model_name, output_dim)
        elif "clip" in model_name.lower():
            self._init_clip(model_name, output_dim)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use T5 or CLIP.")

    def _init_t5(self, model_name, output_dim):
        from transformers import T5EncoderModel, AutoTokenizer
        self._model_type = "t5"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)

        if self.freeze:
            self.encoder.eval()
            self.encoder.requires_grad_(False)

        enc_dim = self.encoder.config.d_model  # t5-small=512, t5-base=768
        self.proj = nn.Linear(enc_dim, output_dim)
        # Null embedding for unconditional / CFG
        self.null_embed = nn.Parameter(torch.zeros(1, 1, output_dim))
        nn.init.normal_(self.null_embed, std=0.02)

    def _init_clip(self, model_name, output_dim):
        from transformers import CLIPTextModel, AutoTokenizer
        self._model_type = "clip"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = CLIPTextModel.from_pretrained(model_name)

        if self.freeze:
            self.encoder.eval()
            self.encoder.requires_grad_(False)

        enc_dim = self.encoder.config.hidden_size  # clip-vit-base=512, large=768
        self.proj = nn.Linear(enc_dim, output_dim)
        self.null_embed = nn.Parameter(torch.zeros(1, 1, output_dim))
        nn.init.normal_(self.null_embed, std=0.02)

    @property
    def lm_dim(self):
        return self.encoder.config.d_model if self._model_type == "t5" \
            else self.encoder.config.hidden_size

    def tokenize(self, texts: list[str], device: torch.device):
        """Tokenize a list of strings. Returns dict with input_ids, attention_mask."""
        if self._model_type == "t5":
            return self.tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(device)
        else:  # clip
            return self.tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(device)

    def forward(self, text_tokens: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode pre-tokenized text.

        Args:
            text_tokens: dict with 'input_ids' and 'attention_mask'
        Returns:
            (cond_tokens, cond_mask):
                cond_tokens: (B, seq_len, output_dim)
                cond_mask: (B, seq_len) bool
        """
        if self.freeze:
            with torch.no_grad():
                hidden = self._encode(text_tokens)
        else:
            hidden = self._encode(text_tokens)

        cond_tokens = self.proj(hidden.float())
        cond_mask = text_tokens["attention_mask"].bool()
        return cond_tokens, cond_mask

    def _encode(self, text_tokens):
        if self._model_type == "t5":
            out = self.encoder(
                input_ids=text_tokens["input_ids"],
                attention_mask=text_tokens["attention_mask"],
            )
            return out.last_hidden_state
        else:  # clip
            out = self.encoder(
                input_ids=text_tokens["input_ids"],
                attention_mask=text_tokens["attention_mask"],
            )
            return out.last_hidden_state

    def encode_texts(self, texts: list[str], device: torch.device):
        """Convenience: tokenize + encode in one call."""
        tokens = self.tokenize(texts, device)
        return self.forward(tokens)

    def get_null_cond(self, batch_size: int, seq_len: int,
                      device: torch.device):
        """Return null conditioning tokens for CFG."""
        return self.null_embed.expand(batch_size, seq_len, -1).to(device)


# ────────────────────────────────────────────────────────────
#  From-scratch CLEVR encoder (for ablation)
# ────────────────────────────────────────────────────────────

CLEVR_COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
CLEVR_SHAPES = ["cube", "sphere", "cylinder"]
CLEVR_SIZES = ["small", "large"]
CLEVR_MATERIALS = ["rubber", "metal"]
CLEVR_RELATIONS = ["left_of", "right_of", "in_front_of", "behind"]
MAX_CLEVR_ENTITIES = 12
MAX_CLEVR_RELATIONS = 30


def clevr_json_to_tensors(cond_json: dict):
    """Convert CLEVR condition JSON -> fixed-size integer tensors."""
    entities = cond_json.get("entities", [])
    relations = cond_json.get("relations", [])

    name_to_idx = {}
    entity_attrs = torch.zeros(MAX_CLEVR_ENTITIES, 4, dtype=torch.long)
    entity_mask = torch.zeros(MAX_CLEVR_ENTITIES, dtype=torch.bool)

    for i, ent in enumerate(entities[:MAX_CLEVR_ENTITIES]):
        name_to_idx[ent["name"]] = i
        attrs = ent["attrs"]
        entity_attrs[i, 0] = CLEVR_COLORS.index(attrs["color"]) if attrs["color"] in CLEVR_COLORS else 0
        entity_attrs[i, 1] = CLEVR_SHAPES.index(attrs["shape"]) if attrs["shape"] in CLEVR_SHAPES else 0
        entity_attrs[i, 2] = CLEVR_SIZES.index(attrs["size"]) if attrs["size"] in CLEVR_SIZES else 0
        entity_attrs[i, 3] = CLEVR_MATERIALS.index(attrs["material"]) if attrs["material"] in CLEVR_MATERIALS else 0
        entity_mask[i] = True

    relation_data = torch.zeros(MAX_CLEVR_RELATIONS, 3, dtype=torch.long)
    relation_mask = torch.zeros(MAX_CLEVR_RELATIONS, dtype=torch.bool)

    for i, rel in enumerate(relations[:MAX_CLEVR_RELATIONS]):
        subj_idx = name_to_idx.get(rel["subj"], 0)
        obj_idx = name_to_idx.get(rel["obj"], 0)
        rel_idx = CLEVR_RELATIONS.index(rel["rel"]) if rel["rel"] in CLEVR_RELATIONS else 0
        relation_data[i] = torch.tensor([subj_idx, rel_idx, obj_idx])
        relation_mask[i] = True

    return entity_attrs, entity_mask, relation_data, relation_mask


class CLEVRTextEncoder(nn.Module):
    """From-scratch CLEVR encoder for ablation comparison."""

    def __init__(self, hidden_size: int, n_transformer_layers: int = 4,
                 n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size

        attr_dim = hidden_size // 4
        self.color_emb = nn.Embedding(len(CLEVR_COLORS), attr_dim)
        self.shape_emb = nn.Embedding(len(CLEVR_SHAPES), attr_dim)
        self.size_emb = nn.Embedding(len(CLEVR_SIZES), attr_dim)
        self.material_emb = nn.Embedding(len(CLEVR_MATERIALS), attr_dim)
        self.entity_proj = nn.Sequential(
            nn.Linear(4 * attr_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.rel_type_emb = nn.Embedding(len(CLEVR_RELATIONS), hidden_size)
        self.relation_proj = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.type_emb = nn.Embedding(2, hidden_size)
        max_seq = MAX_CLEVR_ENTITIES + MAX_CLEVR_RELATIONS
        self.pos_emb = nn.Embedding(max_seq, hidden_size)
        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.null_token, std=0.02)
        self.n_transformer_layers = n_transformer_layers
        if n_transformer_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=n_heads,
                dim_feedforward=hidden_size * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_transformer_layers)
            self.final_norm = nn.LayerNorm(hidden_size)
        else:
            self.transformer = None

    def forward(self, entity_attrs, entity_mask, relation_data, relation_mask):
        B = entity_attrs.shape[0]
        device = entity_attrs.device
        max_E, max_R = entity_attrs.shape[1], relation_data.shape[1]

        e_cat = torch.cat([
            self.color_emb(entity_attrs[:, :, 0]),
            self.shape_emb(entity_attrs[:, :, 1]),
            self.size_emb(entity_attrs[:, :, 2]),
            self.material_emb(entity_attrs[:, :, 3]),
        ], dim=-1)
        e = self.entity_proj(e_cat) * entity_mask.unsqueeze(-1).float()

        subj_feat = torch.gather(e, 1, relation_data[:, :, 0].clamp(0, max_E-1).unsqueeze(-1).expand(-1, -1, self.hidden_size))
        obj_feat = torch.gather(e, 1, relation_data[:, :, 2].clamp(0, max_E-1).unsqueeze(-1).expand(-1, -1, self.hidden_size))
        rel_feat = self.rel_type_emb(relation_data[:, :, 1])
        r = self.relation_proj(torch.cat([subj_feat, rel_feat, obj_feat], dim=-1))
        r = r * relation_mask.unsqueeze(-1).float()

        cond_tokens = torch.cat([e, r], dim=1)
        full_mask = torch.cat([entity_mask, relation_mask], dim=1)
        type_ids = torch.cat([torch.zeros(max_E, dtype=torch.long, device=device),
                              torch.ones(max_R, dtype=torch.long, device=device)])
        pos_ids = torch.arange(max_E + max_R, device=device)
        cond_tokens = cond_tokens + self.type_emb(type_ids) + self.pos_emb(pos_ids)
        null = self.null_token.expand(B, cond_tokens.shape[1], -1)
        cond_tokens = torch.where(full_mask.unsqueeze(-1), cond_tokens, null)
        if self.transformer is not None:
            cond_tokens = self.transformer(cond_tokens, src_key_padding_mask=~full_mask)
            cond_tokens = self.final_norm(cond_tokens)
        return cond_tokens, full_mask


# ────────────────────────────────────────────────────────────
#  Text-Conditioned DiT
# ────────────────────────────────────────────────────────────

class TextConditionedDiT(nn.Module):
    """DiT conditioned on text via cross-attention.

    Two encoder modes:
      1. 'pretrained' (default): Frozen pretrained LM (T5/CLIP) +
         learned projection. JSON → text → LM → project → cross-attn.
      2. 'scratch': From-scratch CLEVRTextEncoder for ablation.

    DiT backbone is identical to MultiResConditionalDiT minus the
    image encoder and multi-resolution hierarchy.
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        vae_downsample_factor: int = 1,
        # --- text encoder ---
        encoder_mode: str = "pretrained",  # "pretrained" or "scratch"
        pretrained_model_name: str = "google-t5/t5-base",
        pretrained_max_length: int = 256,
        freeze_text_encoder: bool = True,
        # scratch encoder params (only if encoder_mode="scratch")
        cond_hidden_size: int = 512,
        cond_n_transformer_layers: int = 4,
        cond_n_heads: int = 8,
        cond_dropout: float = 0.0,
        # --- DiT backbone ---
        dit_patch_size: int = 16,
        dit_hidden_size: int = 768,
        dit_n_heads: int = 12,
        dit_n_blocks: int = 12,
        dit_mlp_ratio: float = 4.0,
        dit_dropout: float = 0.0,
        dit_bottleneck_dim: int = 128,
        dit_in_context_len: int = 0,
        dit_in_context_start: int = 4,
        # --- common ---
        uncond_drop_prob: float = 0.1,
    ):
        super().__init__()

        self.image_size = image_size
        self.vae_downsample_factor = vae_downsample_factor
        self.latent_size = image_size // vae_downsample_factor
        self.uncond_drop_prob = uncond_drop_prob
        self.encoder_mode = encoder_mode

        self.dit_patch_size = dit_patch_size
        self.dit_hidden_size = dit_hidden_size
        self.dit_n_heads = dit_n_heads
        self.dit_n_blocks = dit_n_blocks
        self._in_channels = in_channels
        self.in_context_len = dit_in_context_len
        self.in_context_start = dit_in_context_start

        assert self.latent_size % dit_patch_size == 0
        self.grid_size = self.latent_size // dit_patch_size

        # ── Text Encoder ──
        if encoder_mode == "pretrained":
            self.text_encoder = PretrainedTextEncoder(
                model_name=pretrained_model_name,
                output_dim=dit_hidden_size,
                max_length=pretrained_max_length,
                freeze=freeze_text_encoder,
            )
            self.cond_encoder = None
            self.cond_proj = None
        else:
            self.text_encoder = None
            self.cond_encoder = CLEVRTextEncoder(
                hidden_size=cond_hidden_size,
                n_transformer_layers=cond_n_transformer_layers,
                n_heads=cond_n_heads,
                dropout=cond_dropout,
            )
            self.cond_proj = nn.Linear(cond_hidden_size, dit_hidden_size)

        # ── Patch embedding ──
        self.patch_embed = _BottleneckPatchEmbed(
            self.latent_size, dit_patch_size, in_channels,
            dit_bottleneck_dim, dit_hidden_size,
        )

        # ── Positional embeddings ──
        num_img_tokens = self.grid_size ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_img_tokens, dit_hidden_size),
            requires_grad=False,
        )

        # ── In-context tokens ──
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, dit_hidden_size))
            nn.init.normal_(self.in_context_posemb, std=0.02)

        # ── Timestep embedding ──
        self._t_freq_dim = 256
        self.time_embed = nn.Sequential(
            nn.Linear(self._t_freq_dim, dit_hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(dit_hidden_size, dit_hidden_size, bias=True),
        )

        # ── 2D RoPE ──
        half_head_dim = dit_hidden_size // dit_n_heads // 2
        hw = self.grid_size
        rope_cos, rope_sin = _build_2d_rope(half_head_dim, hw, 0)
        self.register_buffer('_rope_cos', rope_cos)
        self.register_buffer('_rope_sin', rope_sin)
        if self.in_context_len > 0:
            rope_cos_ext, rope_sin_ext = _build_2d_rope(
                half_head_dim, hw, self.in_context_len)
            self.register_buffer('_rope_cos_ext', rope_cos_ext)
            self.register_buffer('_rope_sin_ext', rope_sin_ext)

        # ── Transformer blocks ──
        self.blocks = nn.ModuleList()
        for i in range(dit_n_blocks):
            in_middle = (dit_n_blocks // 4 * 3 > i >= dit_n_blocks // 4)
            a_drop = dit_dropout if in_middle else 0.0
            p_drop = dit_dropout if in_middle else 0.0
            self.blocks.append(
                _DiTBlock(dit_hidden_size, dit_n_heads, dit_mlp_ratio,
                          attn_drop=a_drop, proj_drop=p_drop))

        # ── Final layer ──
        self.final_layer = _DiTFinalLayer(
            dit_hidden_size, in_channels * dit_patch_size ** 2,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Only init DiT weights, not the pretrained LM
        self.patch_embed.apply(_basic_init)
        self.time_embed.apply(_basic_init)
        for block in self.blocks:
            block.apply(_basic_init)
        self.final_layer.apply(_basic_init)
        if self.cond_proj is not None:
            self.cond_proj.apply(_basic_init)
        if self.cond_encoder is not None:
            self.cond_encoder.apply(_basic_init)

        # Fixed sin-cos pos embed
        pos = _get_2d_sincos_pos_embed(self.dit_hidden_size, self.grid_size)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos).float().unsqueeze(0))

        # Timestep MLP
        nn.init.normal_(self.time_embed[0].weight, std=0.02)
        nn.init.normal_(self.time_embed[2].weight, std=0.02)

        # Patch embed
        w1 = self.patch_embed.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.patch_embed.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj2.bias, 0)

        # Zero-out adaLN
        for block in self.blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        G = self.grid_size
        p = self.dit_patch_size
        C = self._in_channels
        x = x.view(B, G, G, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, C, G * p, G * p)

    # ── Pretrained encoder forward ──

    def forward_pretrained(
        self, x_t, t, text_tokens=None, return_uncond=False,
    ):
        """Forward with pretrained text encoder.

        Args:
            x_t: (B, C, H, W) noisy image
            t:   (B,) timestep
            text_tokens: dict from tokenizer (input_ids, attention_mask)
            return_uncond: use null conditioning
        """
        B, device, dtype = x_t.shape[0], x_t.device, x_t.dtype
        K = self.in_context_len

        if return_uncond or text_tokens is None:
            # Use null conditioning — fixed-length null embed
            cond_tokens = self.text_encoder.get_null_cond(B, 1, device).to(dtype)
            xa_mask = None
        else:
            cond_tokens, cond_mask = self.text_encoder(text_tokens)
            cond_tokens = cond_tokens.to(dtype)

            # CFG: randomly drop conditions during training
            if self.training and self.uncond_drop_prob > 0:
                drop = torch.rand(B, device=device) < self.uncond_drop_prob
                null = self.text_encoder.get_null_cond(
                    B, cond_tokens.shape[1], device).to(dtype)
                cond_tokens = torch.where(drop[:, None, None], null, cond_tokens)
                cond_mask = cond_mask & ~drop.unsqueeze(-1)

            xa_mask = cond_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, M)

        return self._dit_forward(x_t, t, cond_tokens, xa_mask)

    # ── Scratch encoder forward ──

    def forward_scratch(
        self, x_t, t, entity_attrs=None, entity_mask=None,
        relation_data=None, relation_mask=None, return_uncond=False,
    ):
        """Forward with from-scratch CLEVRTextEncoder."""
        B, device, dtype = x_t.shape[0], x_t.device, x_t.dtype
        K = self.in_context_len

        if return_uncond or entity_attrs is None:
            max_seq = MAX_CLEVR_ENTITIES + MAX_CLEVR_RELATIONS
            cond_tokens = self.cond_encoder.null_token.expand(B, max_seq, -1)
            cond_tokens = self.cond_proj(cond_tokens.to(dtype))
            xa_mask = None
        else:
            if self.training and self.uncond_drop_prob > 0:
                drop = torch.rand(B, device=device) < self.uncond_drop_prob
            else:
                drop = None

            cond_tokens, cond_valid = self.cond_encoder(
                entity_attrs, entity_mask, relation_data, relation_mask)
            if drop is not None:
                null = self.cond_encoder.null_token.expand(B, cond_tokens.shape[1], -1)
                cond_tokens = torch.where(drop[:, None, None], null, cond_tokens)
                cond_valid = cond_valid & ~drop.unsqueeze(-1)
            cond_tokens = self.cond_proj(cond_tokens.to(dtype))
            xa_mask = cond_valid.unsqueeze(1).unsqueeze(1)

        return self._dit_forward(x_t, t, cond_tokens, xa_mask)

    # ── Shared DiT backbone ──

    def _dit_forward(self, x_t, t, cond_tokens, xa_mask):
        dtype = x_t.dtype
        K = self.in_context_len

        img_tokens = self.patch_embed(x_t) + self.pos_embed
        t_freq = _sinusoidal_timestep_embedding(t, self._t_freq_dim).to(dtype)
        c = self.time_embed(t_freq)

        tokens = img_tokens
        for i, block in enumerate(self.blocks):
            if K > 0 and i == self.in_context_start:
                ic = c.unsqueeze(1).expand(-1, K, -1) + self.in_context_posemb
                tokens = torch.cat([ic, tokens], dim=1)
            has_prefix = (K > 0 and i >= self.in_context_start)
            rope_cos = self._rope_cos_ext if has_prefix else self._rope_cos
            rope_sin = self._rope_sin_ext if has_prefix else self._rope_sin
            tokens = block(tokens, c, cond=cond_tokens,
                           sa_mask=None, xa_mask=xa_mask,
                           rope_cos=rope_cos, rope_sin=rope_sin)

        img_out = tokens[:, K:, :] if K > 0 else tokens
        img_out = self.final_layer(img_out, c)
        return self._unpatchify(img_out)

    # ── Unified forward ──

    def forward(self, x_t, t, text_tokens=None,
                entity_attrs=None, entity_mask=None,
                relation_data=None, relation_mask=None,
                return_uncond=False):
        if self.encoder_mode == "pretrained":
            return self.forward_pretrained(x_t, t, text_tokens, return_uncond)
        else:
            return self.forward_scratch(
                x_t, t, entity_attrs, entity_mask,
                relation_data, relation_mask, return_uncond)

    def describe(self) -> str:
        num_img = self.grid_size ** 2
        K = self.in_context_len
        if self.encoder_mode == "pretrained":
            enc_desc = f"Pretrained LM: {self.text_encoder.model_name} (frozen={self.text_encoder.freeze})"
            enc_params = sum(p.numel() for p in self.text_encoder.encoder.parameters())
            enc_desc += f"\n  LM params: {enc_params/1e6:.1f}M (frozen)"
            proj_params = sum(p.numel() for p in self.text_encoder.proj.parameters())
            enc_desc += f", projection: {proj_params/1e6:.2f}M (trained)"
        else:
            enc_desc = f"CLEVRTextEncoder (scratch, hidden={self.cond_encoder.hidden_size})"
        lines = [
            f"=== TextConditionedDiT (Baseline) ===",
            f"Image: {self.image_size}x{self.image_size}",
            f"Encoder mode: {self.encoder_mode}",
            f"  {enc_desc}",
            f"Backbone: DiT (patch={self.dit_patch_size}, hidden={self.dit_hidden_size}, "
            f"heads={self.dit_n_heads}, blocks={self.dit_n_blocks})",
            f"Cross-attention: full (no spatial masking)",
            f"Image tokens: {self.grid_size}x{self.grid_size} = {num_img}",
            f"In-context: {K} tokens" if K > 0 else "In-context: disabled",
            f"CFG uncond drop: {self.uncond_drop_prob}",
        ]
        return "\n".join(lines)
