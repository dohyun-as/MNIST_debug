import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fsq import FSQ
from Discretizer import FSQDiscretizer, VQDiscretizer
from diffusers import UNet2DConditionModel, UNet2DModel


class PatchImageCondition2DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        feat_channels: int = 128,
        grid_size: int = 9,
        base_channels: int = 64,
        blocks_per_stage: int = 2,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.grid_size = grid_size

        # [1] Stem: 초기 채널 확장
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(inplace=True)
        )

        # [2] Hierarchical Stages: MNIST 한 장을 제대로 훑는 구조
        # Stage 1: 32x32 -> 16x16
        self.layer1 = self._make_layer(base_channels, base_channels * 2, blocks_per_stage, down=True)
        
        # Stage 2: 16x16 -> 8x8
        self.layer2 = self._make_layer(base_channels * 2, base_channels * 4, blocks_per_stage, down=True)
        
        # Stage 3: 8x8 -> 4x4 (숫자의 고수준 특징 추출)
        self.layer3 = self._make_layer(base_channels * 4, base_channels * 4, blocks_per_stage, down=True)

        final_ch = base_channels * 4

        # [3] Final Projection
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(final_ch, feat_channels),
            nn.LayerNorm(feat_channels),
            nn.SiLU(inplace=True)
        )

        # (선택) 2D positional embedding
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, feat_channels, grid_size, grid_size))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def _make_layer(self, c_in, c_out, blocks, down):
        layers = [ResBlock2D(c_in, c_out, down=down)]
        for _ in range(blocks - 1):
            layers.append(ResBlock2D(c_out, c_out, down=False))
        return nn.Sequential(*layers)

    def _pad_to_grid(self, x: torch.Tensor):
        B, C, H, W = x.shape
        Gh = Gw = self.grid_size
        newH = int(math.ceil(H / Gh) * Gh)
        newW = int(math.ceil(W / Gw) * Gw)
        padH, padW = newH - H, newW - W
        if padH > 0 or padW > 0:
            x = F.pad(x, (0, padW, 0, padH), mode="replicate")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_to_grid(x)
        B, C, H, W = x.shape
        Gh = Gw = self.grid_size
        ph, pw = H // Gh, W // Gw

        # 1. Patch 분리: (B*81, C, ph, pw)
        patches = F.unfold(x, kernel_size=(ph, pw), stride=(ph, pw))
        patches = patches.transpose(1, 2).contiguous().view(B * Gh * Gw, C, ph, pw)

        # 2. Deep Encoding
        h = self.stem(patches)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h) # (B*81, final_ch, 4, 4) 수준까지 압축
        
        h = self.pool(h).flatten(1) # (B*81, final_ch)
        h = self.proj(h)            # (B*81, feat_channels)

        # 3. 격자 형태로 재구성: (B, Cfeat, 9, 9)
        feat = h.view(B, Gh * Gw, self.feat_channels).transpose(1, 2).contiguous()
        feat = feat.view(B, self.feat_channels, Gh, Gw)

        if self.use_pos_emb:
            feat = feat + self.pos_emb

        return feat
    
class ResBlock2D(nn.Module):
    def __init__(self, c_in, c_out, down=False, groups=8):
        super().__init__()
        stride = 2 if down else 1
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(min(groups, c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(groups, c_out), c_out)
        self.act = nn.SiLU(inplace=True)
        self.skip = None
        if down or c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride)

    def forward(self, x):
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        s = self.skip(x) if self.skip is not None else x
        return self.act(h + s)

class ImageCondition2DEncoder(nn.Module):
    """
    cond_image: (B, Cin, H, W) -> feat2d: (B, Cfeat, h, w)
    - 여기서는 resize/flatten 절대 안 함
    """
    def __init__(
        self,
        in_channels: int = 1,
        feat_channels: int = 128,       # ← 이 채널이 이후 concat에도, token에도 쓰임
        downsample_factor: int = 16,
        base_channels: int = 64,
        blocks_per_stage: int = 2,
        groups: int = 8,
        channel_cap_mult: int = 8,
    ):
        super().__init__()
        assert downsample_factor > 0 and (downsample_factor & (downsample_factor - 1)) == 0, \
            f"downsample_factor must be power of 2, got {downsample_factor}"
        num_down = int(math.log2(downsample_factor))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(min(groups, base_channels), base_channels),
            nn.SiLU(inplace=True),
        )

        ch = base_channels
        stages = []
        for i in range(num_down):
            ch_next = base_channels * min(2 ** (i + 1), channel_cap_mult)
            blocks = [ResBlock2D(ch, ch_next, down=True, groups=groups)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(ResBlock2D(ch_next, ch_next, down=False, groups=groups))
            stages.append(nn.Sequential(*blocks))
            ch = ch_next
        self.stages = nn.ModuleList(stages)

        # ✅ 최종 2D feature 채널을 feat_channels로 고정
        self.head = nn.Sequential(
            nn.Conv2d(ch, feat_channels, 1),
            nn.GroupNorm(min(groups, feat_channels), feat_channels),
            nn.SiLU(inplace=True),
        )

        self.feat_channels = feat_channels
        self.downsample_factor = downsample_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for stage in self.stages:
            h = stage(h)
        h = self.head(h)  # (B, feat_channels, h, w)
        return h
    

class ConditionalUNet(nn.Module):
    """
    Conditional diffusion model using diffusers.UNet2DConditionModel.

    - x_t: (B, 1, H, W)
    - t:   (B,) timesteps
    - y:   (B,) class labels (0~num_classes-1)

    conditioning 전략:
      * image_conditioning = False:
          -> class embedding(y) 기반 conditioning
      * image_conditioning = True:
          -> (기본) SimpleImageEncoder(cond_image) 기반 conditioning
             (원하면 encoder를 직접 넘겨서 교체 가능)
      * 필요 시 encoder_hidden_states를 직접 넘겨서 사용 가능

    UNet 설정은 unet_config dict로 외부에서 주입 가능:
      - 예: JSON 파일로 저장 후, 로드해서 전달
    """

    def __init__(
        self,
        num_classes: int = 10,
        class_embed_dim: int = 128,
        image_size: int = 32,
        image_conditioning: bool = False,
        encoder: nn.Module | None = None,
        cond_dim: int | None = None,
        cond_in_channels: int = 1,     # image conditioning 시 cond_image 채널 수
        feat_channels: int = 128,
        unet_config: dict | None = None,
        grid_conditioning: bool = False, 
        grid_vocab_size: int = 10, 
        grid_hw: int = 9,
        uncond_drop_prob: float = 0.1,
        # --- concat conditioning 옵션 ---
        concat_conditioning: bool = False,
        concat_downsample_factor: int = 16,
        # ---------------------------------------
        use_fsq: bool = True,
        fsq_levels: list[int] = [8, 8, 8, 5],
        fsq_drop_quant_p: float = 0.0,
        fsq_corrupt_tokens_p: float = 0.0,
        # ---------------------------------------
        vq_codebook_size: int = 9,
        vq_beta: float = 0.25,
        use_vq_discretizer : bool = False,
        # ---------------------------------------
        patch_conditioning: bool = False,
        patch_grid_size: int = 9,
        patch_use_pos_emb: bool = False,
    ):
        super().__init__()

        # 1) cond_dim 결정
        #    - 우선순위: 인자로 온 cond_dim > unet_config["cross_attention_dim"] > class_embed_dim
        if cond_dim is not None:
            self.cond_dim = cond_dim
        elif unet_config is not None and "cross_attention_dim" in unet_config:
            self.cond_dim = unet_config["cross_attention_dim"]
        else:
            self.cond_dim = class_embed_dim

        self.concat_conditioning = concat_conditioning
        self.concat_channels = self.cond_dim
        self.concat_downsample_factor = concat_downsample_factor


        self.grid_conditioning = grid_conditioning
        self.image_conditioning = image_conditioning

        self.use_fsq = use_fsq and image_conditioning  # image_cond에서만 쓰는 걸 추천
        self.use_vq_discretizer = use_vq_discretizer and image_conditioning
        # --- safety check ---
        if self.use_fsq and self.use_vq_discretizer:
            raise ValueError(
                "Both use_fsq and use_vq_discretizer are True. "
                "Choose exactly one discretization method."
            )
        
        self.cond_drop_prob = uncond_drop_prob


        # 2) image conditioning일 때 encoder 설정
        if self.image_conditioning:
            # (B,C,H,W)->(B,D)
            if encoder is None:
                if patch_conditioning:
                    self.encoder = PatchImageCondition2DEncoder(
                        in_channels=cond_in_channels,
                        feat_channels=feat_channels,
                        grid_size=patch_grid_size,
                        use_pos_emb=patch_use_pos_emb,
                    )
                else:
                    self.encoder = ImageCondition2DEncoder(
                        in_channels=cond_in_channels,
                        feat_channels=feat_channels,            # 예: 256
                        downsample_factor=concat_downsample_factor,
                    )
            else:
                self.encoder = encoder

                

            self.cond_token_proj = nn.Linear(self.encoder.feat_channels, self.cond_dim)

            # --- FSQ modules (only if use_fsq) ---
            if self.use_fsq:
                self.discretizer = FSQDiscretizer(
                    slot_dim=self.cond_dim,
                    levels=fsq_levels, 
                    drop_quant_p=fsq_drop_quant_p,
                    corrupt_tokens_p=fsq_corrupt_tokens_p,
                )
            elif self.use_vq_discretizer:
                self.discretizer = VQDiscretizer(
                slot_dim=self.cond_dim,
                codebook_size=vq_codebook_size,        # <- 원하는 vocab 크기 (예: 9, 33, 256, 512, 1024 ...)
                vq_dim=None,              # None이면 slot_dim 그대로 quantize
                beta=vq_beta,              # VQ loss 내에서 사용되는 commitment loss 계수
                use_ema=True,             # 추천
                ema_decay=0.99,
                drop_quant_p=fsq_drop_quant_p,
                corrupt_tokens_p=fsq_corrupt_tokens_p,
                )

                # self.cond_token_proj = None
                
                # fsq_dim = len(fsq_levels)  # d = len(levels)

                # self.fsq_in_proj = nn.Linear(self.encoder.feat_channels, fsq_dim)

                # self.fsq = FSQ(
                #     latents_read_key="z",
                #     quants_write_key="q",
                #     tokens_write_key="tok",
                #     levels=list(fsq_levels),
                #     drop_quant_p=fsq_drop_quant_p,
                #     corrupt_tokens_p=fsq_corrupt_tokens_p,
                #     packed_call=False,  # 여기선 텐서만 forward_z로 쓸 거라
                # )

                # self.post_fsq_proj = nn.Linear(fsq_dim, self.cond_dim)
                    
        elif self.grid_conditioning:
            self.grid_hw = grid_hw
            self.grid_vocab_size = grid_vocab_size
            # 숫자 embedding
            self.grid_embed = nn.Embedding(grid_vocab_size, self.cond_dim)
            # 2D positional embedding
            self.grid_pos_embed = nn.Parameter(
                torch.zeros(1, grid_hw * grid_hw, self.cond_dim)
            )
            nn.init.trunc_normal_(self.grid_pos_embed, std=0.02)

            self.encoder = None
            self.class_embedding = None

        else:
            self.encoder = None
            # 3) label embedding (B,) -> (B, cond_dim)
            self.class_embedding = nn.Embedding(num_classes, self.cond_dim)

        # 4) UNet2DConditionModel 설정
        unet_config = dict(unet_config)

        if self.concat_conditioning:
            unet_config["in_channels"] = unet_config["in_channels"] + self.concat_channels
            self.cond_up = nn.Sequential(
            # NOTE: we will call with either scale_factor or target size in forward logic
            nn.Conv2d(self.cond_dim, self.cond_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, self.cond_dim), self.cond_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.cond_dim, self.cond_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, self.cond_dim), self.cond_dim),
            nn.SiLU(inplace=True),
            )       
        else:
            self.cond_up = None

        if self.cond_drop_prob > 0:
            if self.image_conditioning:
                # image_size 기반으로 "기대 토큰 수" K 계산
                h = image_size // self.concat_downsample_factor
                w = image_size // self.concat_downsample_factor
                null_k = max(1, h * w)
            elif self.grid_conditioning:
                null_k = self.grid_hw * self.grid_hw
            else:
                null_k = 1

            self.null_cond = nn.Parameter(torch.zeros(1, null_k, self.cond_dim))
            print("cond_token_num", null_k)
            print("self.cond_dim_null",self.cond_dim)
            nn.init.normal_(self.null_cond, std=0.02)
        else:
            self.null_cond = None

        if concat_conditioning:
            self.unet = UNet2DModel(**unet_config)
        else:
            self.unet = UNet2DConditionModel(**unet_config)

    def _get_null_tokens(self, B: int, L: int, device, dtype):
        """
        null_cond: (1, K, D) 에서 L만큼 slice해서 (B, L, D) 생성
        """
        if self.null_cond is None:
            raise RuntimeError("cond_drop_prob==0 인데 null tokens 요청됨")
        # K가 L보다 작으면(입력 해상도 바뀐 경우) 마지막 토큰으로 패딩
        K = self.null_cond.shape[1]
        if K >= L:
            base = self.null_cond[:, :L, :]
        else:
            pad = self.null_cond[:, -1:, :].expand(1, L - K, self.cond_dim)
            base = torch.cat([self.null_cond, pad], dim=1)
        return base.to(device=device, dtype=dtype).expand(B, L, self.cond_dim)
    
    # def cond_encoding(
    #     self,
    #     y: torch.Tensor | None = None,
    #     cond_image: torch.Tensor | None = None,
    #     encoder_hidden_states: torch.Tensor | None = None,
    #     grid: torch.Tensor | None = None,
    # ) -> torch.Tensor:
    #     # 1) 이미 인코딩된 상태가 들어온 경우
    #     if encoder_hidden_states is not None:
    #         return encoder_hidden_states 

    #     if self.image_conditioning:
    #         if cond_image is None:
    #             raise ValueError("image_conditioning=True 인데 cond_image가 없음")
    #         return self.encoder(cond_image)  # (B,Cfeat,h,w) 그대로


    #     if self.grid_conditioning:
    #         if grid is None:
    #             raise ValueError("grid_conditioning=True 인데 grid가 없음")
    #         g = grid.to(torch.long)

    #         emb = self.grid_embed(g)  # (B,H,W,D)
    #         B, H, W, D = emb.shape
    #         tokens = emb.view(B, H * W, D)
    #         pos = self.grid_pos_embed[:, : H * W, :]
    #         return tokens + pos  # (B,L,D)

        
    #     if y is None:
    #         raise ValueError("class conditioning 모드에서는 y가 필요합니다.")
    #     return self.class_embedding(y)
    def cond_encoding(
        self,
        y=None,
        cond_image=None,
        encoder_hidden_states=None,
        grid=None,
        return_token_ids: bool = False,
        return_uncond: bool= False,
    ):
        tok_ids = None

        # 0) tokens 만들기
        if encoder_hidden_states is not None:
            tokens = encoder_hidden_states  # (B,L,D)
            return tokens

        elif self.image_conditioning:
            if cond_image is None:
                raise ValueError("image_conditioning=True 인데 cond_image가 없음")
            tokens, _ = self.encode_image_to_tokens(cond_image)  # (B,L,D), (B,h,w) or None

        elif self.grid_conditioning:
            if grid is None:
                raise ValueError("grid_conditioning=True 인데 grid가 없음")
            g = grid.to(torch.long)
            emb = self.grid_embed(g)  # (B,H,W,D)
            B, H, W, D = emb.shape
            tokens = emb.view(B, H * W, D)
            pos = self.grid_pos_embed[:, : H * W, :]
            tokens = tokens + pos  # (B,L,D)

        else:
            if y is None:
                raise ValueError("class conditioning 모드에서는 y가 필요합니다.")
            tokens = self.class_embedding(y)[:, None, :]  # (B,1,D)

        if self.use_fsq or self.use_vq_discretizer:
            tokens, tok_ids = self.discretizer(tokens)

        # ✅ eval CFG용: unconditional 강제
        if return_uncond:
            # ✅ 여기서 null_tokens 정의
            B, L, D = tokens.shape
            null_tokens = self._get_null_tokens(B, L, device=tokens.device, dtype=tokens.dtype)
            tokens = null_tokens
            # if tok_ids is not None:
            #     tok_ids = torch.zeros_like(tok_ids)
            return (tokens, tok_ids) if return_token_ids else tokens
    
        # 1) ✅ training이면 10% 확률로 unconditional 토큰으로 치환
        if self.training and self.cond_drop_prob > 0.0:
            B, L, D = tokens.shape
            null_tokens = self._get_null_tokens(B, L, device=tokens.device, dtype=tokens.dtype)

            drop = (torch.rand(B, 1, 1, device=tokens.device) < self.cond_drop_prob)  # (B,1,1)
            tokens = torch.where(drop, null_tokens, tokens)

            # (선택) image 토큰 id도 같이 무력화
            # if tok_ids is not None:
            #     tok_ids = torch.where(drop.view(B, 1, 1).expand_as(tok_ids), torch.zeros_like(tok_ids), tok_ids)

        return (tokens, tok_ids) if return_token_ids else tokens

    # def encode_image_to_tokens(self, cond_image: torch.Tensor):
    #     # print("cond_image", cond_image.shape)
    #     feat2d = self.encoder(cond_image)  # (B,Cfeat,h,w)
    #     # print("feat2d", feat2d.shape)
    #     feat_tok = feat2d.permute(0, 2, 3, 1).contiguous()  # (B,h,w,Cfeat)
    #     # print("feat_tok", feat_tok.shape)

    #     tok_ids = None
    #     if self.use_fsq:
    #         z = self.fsq_in_proj(feat_tok)      # (B,h,w,fsq_dim)
    #         q, tok_ids = self.fsq.forward_z(z)  # (B,h,w,fsq_dim), (B,h,w)
    #         tok2d = self.post_fsq_proj(q)       # (B,h,w,cond_dim)
    #     else:
    #         tok2d = self.cond_token_proj(feat_tok)  # (B,h,w,cond_dim)

    #     # print("tok2d", tok2d.shape)
    #     B, h, w, D = tok2d.shape
    #     tokens = tok2d.view(B, h * w, D)  # (B,L,D)
    #     # print("tokens", tokens.shape)
    #     return tokens, tok_ids, (h, w)
    def encode_image_to_tokens(self, cond_image: torch.Tensor):
        # print("cond_image", cond_image.shape)
        feat2d = self.encoder(cond_image)  # (B,Cfeat,h,w)
        # print("feat2d", feat2d.shape)
        feat_tok = feat2d.permute(0, 2, 3, 1).contiguous()  # (B,h,w,Cfeat)
        # print("feat_tok", feat_tok.shape)

        tok2d = self.cond_token_proj(feat_tok)  # (B,h,w,cond_dim)

        # print("tok2d", tok2d.shape)
        B, h, w, D = tok2d.shape
        tokens = tok2d.view(B, h * w, D)  # (B,L,D)
        # print("tokens", tokens.shape)
        return tokens, (h, w)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        cond_image: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        grid: torch.Tensor | None = None,
        return_aux_loss: bool = False,
    ) -> torch.Tensor:
        # cond_encoding은 "최종 tokens"를 리턴: (B, L, D)
        cond_tokens = self.cond_encoding(
            y=y,
            cond_image=cond_image,
            encoder_hidden_states=encoder_hidden_states,
            grid=grid,
            return_token_ids=False,
        )  # (B, L, D)

        if self.use_vq_discretizer:
            aux_loss = getattr(self.discretizer, "last_vq_loss", None)
            
        x_in = x_t

        # ✅ concat conditioning이면: tokens -> (B, D, h, w) -> upsample -> concat
        if self.concat_conditioning:
            B, L, D = cond_tokens.shape

            if self.image_conditioning:
                if cond_image is None:
                    raise ValueError("concat_conditioning=True 인데 cond_image가 없음")

                Hc, Wc = cond_image.shape[-2], cond_image.shape[-1]

                # encoder가 만든 feature map의 spatial size는 입력/다운샘플 팩터로 결정됨
                if hasattr(self.encoder, "grid_size"):  # patch encoder
                    h = w = self.encoder.grid_size
                else:
                    h = Hc // self.concat_downsample_factor
                    w = Wc // self.concat_downsample_factor
            else:
                # grid_conditioning
                h = w = self.grid_hw

            assert L == h * w, f"L={L} != h*w={h*w} (h={h}, w={w})"

            cond_2d = cond_tokens.view(B, h, w, D).permute(0, 3, 1, 2).contiguous()  # (B,D,h,w)
            cond_2d = F.interpolate(cond_2d, size=x_t.shape[-2:], mode="nearest")    # (B,D,H,W)

            if self.cond_up is not None:
                cond_2d = self.cond_up(cond_2d)  # (B,D,H,W)


            x_in = torch.cat([x_t, cond_2d], dim=1)  # (B, 1+D, H, W)

        # print("x_in", x_in.shape)
        # print("cond_statesc",unet_cond_states.shape)

        if self.concat_conditioning:
            out = self.unet(sample=x_in, timestep=t)
        else:
            out = self.unet(sample=x_in, timestep=t, encoder_hidden_states=cond_tokens)

        pred = out.sample

        if return_aux_loss:
            return pred, aux_loss
        else:
            return pred
