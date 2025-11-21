import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class SimpleImageEncoder(nn.Module):
    """
    간단한 CNN 기반 image encoder.
    (B, C, H, W) -> (B, cond_dim)
    """
    def __init__(self, in_channels: int = 1, cond_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, cond_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x)               # (B, 128, 1, 1)
        h = h.view(h.size(0), -1)     # (B, 128)
        h = self.proj(h)              # (B, cond_dim)
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
        unet_config: dict | None = None,
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

        self.image_conditioning = image_conditioning

        # 2) image conditioning일 때 encoder 설정
        if self.image_conditioning:
            if encoder is None:
                self.encoder = SimpleImageEncoder(
                    in_channels=cond_in_channels,
                    cond_dim=self.cond_dim,
                )
            else:
                self.encoder = encoder
        else:
            self.encoder = None

        # 3) label embedding (B,) -> (B, cond_dim)
        self.class_embedding = nn.Embedding(num_classes, self.cond_dim)

        # 4) UNet2DConditionModel 설정
        if unet_config is None:
            # 기본 config
            unet_config = {
                "sample_size": image_size,
                "in_channels": 1,
                "out_channels": 1,
                "layers_per_block": 2,
                "block_out_channels": (64, 128, 256, 256),
                "down_block_types": (
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                ),
                "up_block_types": (
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                "cross_attention_dim": self.cond_dim,
                "attention_head_dim": 4,
            }
        else:
            # 외부 config에서 일부 필드 없으면 채워주기
            unet_config = dict(unet_config)  # 얕은 복사해서 수정
            # sample_size가 없으면 image_size로
            unet_config.setdefault("sample_size", image_size)
            unet_config.setdefault("in_channels", 1)
            unet_config.setdefault("out_channels", 1)
            # cross_attention_dim은 cond_dim과 일치시키기
            unet_config.setdefault("cross_attention_dim", self.cond_dim)
            if unet_config["cross_attention_dim"] != self.cond_dim:
                # cond_dim을 맞춰줌
                self.cond_dim = unet_config["cross_attention_dim"]

        self.unet = UNet2DConditionModel(**unet_config)

    def cond_encoding(
        self,
        y: torch.Tensor | None = None,
        cond_image: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        조건을 UNet이 바로 쓸 수 있는 encoder_hidden_states (B, L, D)로 변환.

        우선순위:
          1) encoder_hidden_states 가 주어지면 그대로 사용
          2) image_conditioning=True 이고 cond_image 가 있을 때:
               h = encoder(cond_image)
          3) 그 외에는 라벨 y로 class embedding 사용

        반환:
          encoder_hidden_states: (B, L, cond_dim)
        """
        # 1) 이미 인코딩된 상태가 들어온 경우
        if encoder_hidden_states is not None:
            h = encoder_hidden_states

        # 2) image conditioning 모드
        elif self.image_conditioning:
            if cond_image is None:
                raise ValueError(
                    "image_conditioning=True 인 경우 cond_image 를 전달해야 합니다."
                )
            # encoder: (B, C, H, W) -> (B, D) 또는 (B, L, D)
            h = self.encoder(cond_image)

        # 3) class conditioning 모드
        else:
            if y is None:
                raise ValueError(
                    "class conditioning 모드에서는 y (class labels)가 필요합니다."
                )
            h = self.class_embedding(y)  # (B, cond_dim) 또는 (B, D)

        # shape 정규화: (B, D) -> (B, 1, D)
        if h.dim() == 2:
            h = h.unsqueeze(1)
        elif h.dim() == 3:
            # (B, L, D) 그대로
            pass
        else:
            raise ValueError(
                f"Condition encoding must have shape (B, D) or (B, L, D), got {h.shape}"
            )

        return h  # (B, L, cond_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        cond_image: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x_t: (B, 1, H, W)
        t:   (B,) 또는 scalar-like
        y:   (B,)  - class conditioning 모드에서 사용
        cond_image: (B, C, H, W) - image conditioning 모드에서 사용
        encoder_hidden_states: (B, L, D) - 이미 인코딩된 condition 직접 사용

        return: predicted noise, same shape as x_t
        """
        cond_states = self.cond_encoding(
            y=y,
            cond_image=cond_image,
            encoder_hidden_states=encoder_hidden_states,
        )

        out = self.unet(
            sample=x_t,
            timestep=t,
            encoder_hidden_states=cond_states,
        )
        return out.sample
