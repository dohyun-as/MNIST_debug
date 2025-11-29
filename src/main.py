# train_mnist_diffusion.py

import argparse
import os
import json
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed, tqdm  # tqdm from accelerate
from accelerate.utils import ProjectConfiguration

from diffusers import DDPMScheduler, DDIMScheduler
from torchvision.utils import make_grid, save_image

from dataset import MNISTConditionalDataset

from SRM.datasets import DatasetCfg, get_dataset, get_dataset_class
from SRM.type_extensions import ConditioningCfg
from omegaconf import OmegaConf

from model import ConditionalUNet
from sampling import sample_ddim_with_cfg

from concurrent.futures import ThreadPoolExecutor
import importlib

def get_obj_from_str(string, reload=False):
    """Get object from string path."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """Instantiate an object from a config dictionary."""
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

@torch.no_grad()
def vae_encode(vae, x):
    out = vae.encode(x)
    if hasattr(out, "latent_dist"):
        out = out.latent_dist
    return out.sample()

@torch.no_grad()
def vae_decode(vae, z):
    out = vae.decode(z)
    if hasattr(out, "sample"):
        out = out.sample
    return out

def decode_in_chunks(vae, latents, max_batch=4):
    """(B, C, H, W) → VAE decode를 작은 chunk로 나눠서 OOM 방지"""
    outs = []
    B = latents.size(0)
    for i in range(0, B, max_batch):
        chunk = latents[i:i+max_batch]
        outs.append(vae_decode(vae, chunk))
    return torch.cat(outs, dim=0)

def parse_args():
    parser = argparse.ArgumentParser(
        description="MNIST Conditional Diffusion Model Training (Accelerate + diffusers DDPMScheduler)"
    )

    # basic
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to MNIST data root.")
    parser.add_argument("--output_dir", type=str, default="./outputs/mnist_diffusion",
                        help="Directory to save checkpoints and logs.")
    parser.add_argument("--seed", type=int, default=42)

    # dataset
    parser.add_argument(
        "--resize_image_size",
        type=int,
        default=32,
        help="Resize MNIST images to this size before padding."
    )
    parser.add_argument(
        "--pad_image_size",
        type=int,
        default=None,
        help="Optional final canvas size. If set, image is padded to (pad_image_size, pad_image_size)."
    )
    
    parser.add_argument(
        "--sudoku_config",
        type=str,
        default=None,
        help="Optional dataset config JSON (with `target` and `params`). "
             "If None, use MNISTConditionalDataset.",
    )
    parser.add_argument(
        "--grid_hw",
        type=int,
        default=9,
        help="Sudoku size"
    )

    # 🔹 VAE config (옵션: latent-space 학습할 때만 사용)
    parser.add_argument(
        "--vae_config",
        type=str,
        default=None,
        help="Path to VAE config JSON file. If None, train directly on pixels.",
    )
    parser.add_argument(
        "--vae_test",
        action="store_true",
        help="If set, run a VAE reconstruction test (GT vs recon) and exit.",
    )


    # model / unet config
    parser.add_argument("--unet_config", type=str, required=True,
                        help="Path to UNet2DConditionModel config JSON file.")
    
    parser.add_argument(
        "--image_conditioning",
        action="store_true",
        help="If set, use image encoder for conditioning instead of class labels.",
    )

    # training (step 기반)
    parser.add_argument("--max_train_steps", type=int, default=50000,
                        help="Total number of optimizer update steps.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    # diffusion (for DDPMScheduler)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_schedule", type=str, default="linear",
                        choices=["linear", "scaled_linear", "squaredcos_cap_v2"])

    # logging / saving / eval (전부 step 기준)
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save checkpoint every N optimizer steps.")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log every N optimizer steps.")
    parser.add_argument("--eval_every", type=int, default=5000,
                        help="Run evaluation every N optimizer steps.")

    # eval 설정
    parser.add_argument("--eval_num_steps", type=int, default=50,
                        help="Number of diffusion steps for eval sampling.")
    parser.add_argument("--eval_num_samples_per_class", type=int, default=8,
                        help="How many samples to generate per class during eval.")

    # accelerate 관련 옵션
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Accelerate mixed precision mode.",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        help='Accelerate tracker backend, e.g. "tensorboard" or "wandb".',
    )

    args = parser.parse_args()
    return args


def load_vae_from_config(vae_config_path: str, device: torch.device, dtype: torch.dtype):
    """
    JSON 파일로부터 VAE config를 읽어서 AutoencoderKL 생성.
    config 예시 (예전에 쓰던 YAML params 부분만 JSON으로 옮긴 형태):

    {
      "embed_dim": 16,
      "ckpt_path": "/path/to/epoch=...ckpt",
      "ddconfig": {
        "double_z": true,
        "z_channels": 16,
        "resolution": 288,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 128,
        "ch_mult": [1,1,2,2,4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0
      }
    }
    """
    with open(vae_config_path, "r") as f:
        cfg = json.load(f)

    ckpt_path = cfg.pop("ckpt_path", None)

    # AutoencoderKL(**cfg)  # embed_dim, ddconfig 등 사용
    vae = instantiate_from_config(cfg).to(device=device, dtype=torch.float32) # encode/decode는 fp32 권장

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        # Lightning ckpt면 보통 "state_dict" 키 안에 들어 있음
        if "state_dict" in state:
            vae.load_state_dict(state["state_dict"], strict=False)
        else:
            vae.load_state_dict(state, strict=False)

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    return vae

@torch.no_grad()
def run_vae_recon_test(
    accelerator: Accelerator,
    vae,
    train_dataset,
    args,
):
    """
    VAE encode/decode sanity check.
    - MNIST 이미지 몇 개 뽑아서
      [GT | recon] 형태로 붙여서 한 장으로 저장하고 종료.
    """
    if vae is None:
        raise ValueError("--vae_test 를 쓰려면 --vae_config 도 같이 줘야 합니다.")

    if not accelerator.is_main_process:
        return

    device = accelerator.device
    vae.to(device)
    vae.eval()

    accelerator.print("[VAE TEST] Running VAE reconstruction test...")

    class_ids = list(range(1, 10))   # [1,2,3,4,5,6,7,8,9]
    num_classes = len(class_ids)
    n_per_class = args.eval_num_samples_per_class  # 그냥 eval 설정 재사용
    total_b = num_classes * n_per_class

    class_counts = {c: 0 for c in class_ids}
    ref_images = []
    ref_labels = []

    # train_dataset에서 class-wise로 뽑기 (MNISTConditionalDataset 기준)
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]

        if isinstance(sample, dict):
            # SRM sudoku: {"image": (1,H,W), "grid": (Hc,Wc), ...}
            img  = sample["image"]
            grid = sample.get("grid", None)

            if grid is None:
                continue  # grid 없으면 class 정보 없다고 보고 건너뜀

            Hc, Wc = grid.shape[-2], grid.shape[-1]
            cy, cx = Hc // 2, Wc // 2
            lab_val = grid[cy, cx]
            lab_int = int(lab_val)
        else:
            # MNIST: (image, label)
            img, lab = sample
            lab_int = int(lab)

        if lab_int not in class_counts:
            continue

        if class_counts[lab_int] < n_per_class:
            ref_images.append(img.unsqueeze(0))  # (1, C, H, W)
            ref_labels.append(lab_int)
            class_counts[lab_int] += 1

            if sum(class_counts.values()) == total_b:
                break

    if len(ref_images) == 0:
        raise RuntimeError("[VAE TEST] Could not collect any images from dataset.")

    if len(ref_images) < total_b:
        accelerator.print(
            f"[VAE TEST] Warning: only {len(ref_images)}/{total_b} images collected."
        )

    ref_images = torch.cat(ref_images, dim=0).to(device)  # (B, C, H, W)
    B = ref_images.size(0)

    # VAE encode/decode (여기서는 scaling_factor 안 건드림: 순수 재구성 테스트)
    imgs_in = ref_images.to(torch.float32)
    z = vae_encode(vae, imgs_in)              # (B, z_channels, H', W')
    recon = decode_in_chunks(vae, z, max_batch=4)  # (B, C, H, W)

    # [-1,1] → [0,1]
    real_01 = (imgs_in.clamp(-1.0, 1.0) + 1.0) * 0.5
    real_01 = real_01.clamp(0.0, 1.0)

    recon_01 = (recon.clamp(-1.0, 1.0) + 1.0) * 0.5
    recon_01 = recon_01.clamp(0.0, 1.0)

    # [real | recon] 가로로 붙이고 패딩
    pair_imgs = []
    for i in range(B):
        real = real_01[i]   # (C, H, W)
        rec = recon_01[i]   # (C, H, W)

        pair = torch.cat([real, rec], dim=2)  # (C, H, 2W)
        pair = F.pad(pair, (1, 1, 1, 1), value=1.0)  # 테두리 흰색
        pair_imgs.append(pair.unsqueeze(0))

    pair_imgs = torch.cat(pair_imgs, dim=0)  # (B, C, H', W')

    # class별로 한 줄씩: nrow = n_per_class
    nrow = n_per_class
    grid = make_grid(pair_imgs, nrow=nrow, padding=2)

    vae_dir = os.path.join(args.output_dir, "vae_test")
    os.makedirs(vae_dir, exist_ok=True)
    out_path = os.path.join(vae_dir, "vae_recon.png")
    save_image(grid, out_path)

    accelerator.print(f"[VAE TEST] Saved VAE recon grid to {out_path}")

@torch.no_grad()
def run_evaluation(
    accelerator: Accelerator,
    model: ConditionalUNet,
    noise_scheduler: DDPMScheduler,
    unet_config: dict,
    args,
    global_step: int,
    vae=None,
    train_dataset: MNISTConditionalDataset | None = None,
):
    if not accelerator.is_main_process:
        return

    accelerator.print(f"[Eval] Running evaluation at step {global_step}...")

    unwrapped_model = accelerator.unwrap_model(model)
    device = accelerator.device

    # 어떤 conditioning 모드인지 확인
    use_image_cond = getattr(unwrapped_model, "image_conditioning", False)
    use_grid_cond  = getattr(unwrapped_model, "grid_conditioning", False)

    if use_image_cond or use_grid_cond:
        assert train_dataset is not None, "image/grid conditioning에서는 train_dataset이 필요합니다."

    # DDIM scheduler 준비
    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)

    class_ids = list(range(1, 10))   # [1,...,9]
    n_per_class = args.eval_num_samples_per_class
    total_b = len(class_ids) * n_per_class

    # ---------- 1) dataset에서 reference batch 모으기 ----------
    class_counts = {c: 0 for c in class_ids}
    ref_images = []
    ref_grids = []
    ref_labels = []

    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]

        if isinstance(sample, dict):
            # SRM sudoku: {"image": (1,H,W), "grid": (Hc,Wc), ...}
            img  = sample["image"]
            grid = sample.get("grid", None)

            if use_grid_cond and grid is None:
                # grid-conditioning인데 grid가 없으면 쓸 수 없음
                continue

            if grid is not None:
                Hc, Wc = grid.shape[-2], grid.shape[-1]
                cy, cx = Hc // 2, Wc // 2
                lab_int = int(grid[cy, cx])
            else:
                # MNIST처럼 dict를 썼지만 grid 없는 경우라면 라벨 정보 없음 → 스킵
                continue
        else:
            # MNIST: (img, label)
            img, lab = sample
            grid = None
            lab_int = int(lab)

        if lab_int not in class_counts:
            continue
        if class_counts[lab_int] >= n_per_class:
            continue

        ref_images.append(img.unsqueeze(0))        # (1, C, H, W)
        if use_grid_cond and grid is not None:
            ref_grids.append(grid.unsqueeze(0))    # (1, Hc, Wc)
        ref_labels.append(lab_int)
        class_counts[lab_int] += 1

        if sum(class_counts.values()) == total_b:
            break

    if len(ref_images) == 0:
        raise RuntimeError("[Eval] reference 이미지를 하나도 못 모았습니다.")

    if len(ref_images) < total_b:
        accelerator.print(
            f"[Eval] Warning: only {len(ref_images)}/{total_b} reference samples collected."
        )

    ref_images = torch.cat(ref_images, dim=0).to(device)   # (B, C, H, W)
    B = ref_images.size(0)

    if use_grid_cond:
        ref_grids = torch.cat(ref_grids, dim=0).to(device)  # (B, Hc, Wc)
    else:
        ref_grids = None

    if not use_image_cond and not use_grid_cond:
        labels = torch.tensor(ref_labels, device=device, dtype=torch.long)  # (B,)
    else:
        labels = None

    # ---------- 2) conditioning 인자 한 군데로 모으기 ----------
    cond_kwargs = {}
    if use_image_cond:
        cond_kwargs["cond_image"] = ref_images
    elif use_grid_cond:
        cond_kwargs["grid"] = ref_grids
    else:
        cond_kwargs["y"] = labels

    # ---------- 3) DDIM sampling (모든 모드 공통) ----------
    in_channels = unet_config.get("in_channels", 1)
    sample_size = unet_config.get("sample_size", ref_images.shape[-1])

    ddim_scheduler.set_timesteps(args.eval_num_steps, device=device)
    x = torch.randn(
        (B, in_channels, sample_size, sample_size),
        device=device,
    )

    for t in tqdm(ddim_scheduler.timesteps, disable=True):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        eps = unwrapped_model(x, t_batch, **cond_kwargs)
        x = ddim_scheduler.step(eps, t, x).prev_sample

    # VAE decode (있으면 latent → pixel)
    if vae is not None:
        scaling_factor = getattr(vae, "scaling_factor", 1.0)
        latents = x / scaling_factor
        imgs = decode_in_chunks(vae, latents, max_batch=4)
    else:
        imgs = x

    # ---------- 4) [real | fake] 그리드 만들기 (공통) ----------
    real_imgs_01 = (ref_images.clamp(-1.0, 1.0) + 1.0) * 0.5
    real_imgs_01 = real_imgs_01.clamp(0.0, 1.0)

    fake_imgs_01 = (imgs.clamp(-1.0, 1.0) + 1.0) * 0.5
    fake_imgs_01 = fake_imgs_01.clamp(0.0, 1.0)

    pair_imgs = []
    for i in range(B):
        real = real_imgs_01[i]  # (C, H, W)
        fake = fake_imgs_01[i]  # (C, H, W)

        pair = torch.cat([real, fake], dim=2)  # (C, H, 2W)
        pair = F.pad(pair, (1, 1, 1, 1), value=1.0)
        pair_imgs.append(pair.unsqueeze(0))

    pair_imgs = torch.cat(pair_imgs, dim=0)
    grid = make_grid(pair_imgs, nrow=n_per_class, padding=2)

    # ---------- 5) 저장 ----------
    eval_dir = os.path.join(args.output_dir, "eval_samples")
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, f"step_{global_step}.png")
    save_image(grid, out_path)

    accelerator.print(f"[Eval] Saved eval samples to {out_path}")

# @torch.no_grad()
# def run_evaluation(
#     accelerator: Accelerator,
#     model: ConditionalUNet,
#     noise_scheduler: DDPMScheduler,
#     unet_config: dict,
#     args,
#     global_step: int,
#     vae=None,
#     train_dataset: MNISTConditionalDataset | None = None,
# ):
#     """
#     eval 시점마다 DDIM 샘플러로 class-wise 이미지 생성.

#     - image_conditioning = False:
#         기존처럼 label 기반 class-wise 샘플링
#     - image_conditioning = True:
#         dataset에서 class별로 이미지 n_per_class개 뽑아서
#         cond_image 로 넣고, [real | generated] 비교 이미지 저장
#     """
#     if not accelerator.is_main_process:
#         return

#     accelerator.print(f"[Eval] Running evaluation at step {global_step}...")

#     unwrapped_model = accelerator.unwrap_model(model)
#     device = accelerator.device

#     # DDIM scheduler 준비
#     ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)

#     class_ids = list(range(1, 10))   # [1,...,9]
#     num_classes = len(class_ids)
#     n_per_class = args.eval_num_samples_per_class
#     total_b = num_classes * n_per_class

#     # ----------------------------------------------------
#     # 1) IMAGE CONDITIONING인 경우: dataset에서 ref image 뽑기
#     # ----------------------------------------------------
#     unwrapped_model = accelerator.unwrap_model(model)
#     if getattr(unwrapped_model, "image_conditioning", False):
#         assert train_dataset is not None, "image_conditioning=True 인 경우 train_dataset이 필요합니다."

#         class_counts = {c: 0 for c in class_ids}
#         ref_images = []
#         ref_labels = []

#         for idx in range(len(train_dataset)):
#             sample = train_dataset[idx]

#             if isinstance(sample, dict):
#                 # SRM sudoku
#                 img  = sample["image"]
#                 grid = sample.get("grid", None)

#                 if grid is None:
#                     continue

#                 Hc, Wc = grid.shape[-2], grid.shape[-1]
#                 cy, cx = Hc // 2, Wc // 2
#                 lab_val = grid[cy, cx]
#                 lab_int = int(lab_val)
#             else:
#                 # MNIST
#                 img, lab = sample
#                 lab_int = int(lab)

#             if lab_int not in class_counts:
#                 continue

#             if class_counts[lab_int] < n_per_class:
#                 ref_images.append(img.unsqueeze(0))  # (1, 1, H, W)
#                 ref_labels.append(lab_int)
#                 class_counts[lab_int] += 1

#                 if sum(class_counts.values()) == total_b:
#                     break

#         if len(ref_images) < total_b:
#             accelerator.print(
#                 f"[Eval] Warning: dataset이 부족해서 {len(ref_images)}/{total_b}개만 사용합니다."
#             )

#         ref_images = torch.cat(ref_images, dim=0).to(device)           # (B, 1, H, W)
#         ref_labels = torch.tensor(ref_labels, device=device, dtype=torch.long)  # (B,)

#         B = ref_images.size(0)

#         # UNet/latent shape 결정
#         in_channels = unet_config.get("in_channels", 1)
#         sample_size = unet_config.get("sample_size", ref_images.shape[-1])

#         # DDIM 타임스텝 설정
#         ddim_scheduler.set_timesteps(args.eval_num_steps, device=device)

#         # 초기 노이즈 (latent 또는 pixel)
#         x = torch.randn(
#             (B, in_channels, sample_size, sample_size),
#             device=device,
#         )

#         # DDIM loop (CFG 없이, guidance_scale=1)
#         for t in tqdm(ddim_scheduler.timesteps, disable=True):
#             t_batch = torch.full((B,), t, device=device, dtype=torch.long)

#             # 이미지 기반 conditioning
#             eps = unwrapped_model(
#                 x,
#                 t_batch,
#                 cond_image=ref_images,   # encoder(cond_image) 사용
#             )

#             step = ddim_scheduler.step(eps, t, x)
#             x = step.prev_sample

#         # latent → pixel (vae 있는 경우)
#         if vae is not None:
#             scaling_factor = getattr(vae, "scaling_factor", 1.0)
#             latents = x / scaling_factor
#             decoded = decode_in_chunks(vae, latents, max_batch=4)
#             imgs = decoded
#         else:
#             imgs = x  # pixel [-1,1]로 가정

#         # real / fake 둘 다 [0,1]로 변환
#         real_imgs_01 = (ref_images.clamp(-1.0, 1.0) + 1.0) * 0.5
#         real_imgs_01 = real_imgs_01.clamp(0.0, 1.0)

#         fake_imgs_01 = (imgs.clamp(-1.0, 1.0) + 1.0) * 0.5
#         fake_imgs_01 = fake_imgs_01.clamp(0.0, 1.0)

#         # [real | fake]로 가로 concat + 테두리 (pad)
#         pair_imgs = []
#         for i in range(B):
#             real = real_imgs_01[i]  # (C, H, W)
#             fake = fake_imgs_01[i]  # (C, H, W)

#             pair = torch.cat([real, fake], dim=2)  # (C, H, 2W)
#             # 테두리: 양쪽 1픽셀씩 흰색 (value=1.0)
#             pair = F.pad(pair, (1, 1, 1, 1), value=1.0)  # (C, H+2, 2W+2)
#             pair_imgs.append(pair.unsqueeze(0))

#         pair_imgs = torch.cat(pair_imgs, dim=0)  # (B, C, H', W')

#         # grid: class별로 한 줄씩 → nrow = n_per_class
#         grid = make_grid(pair_imgs, nrow=n_per_class, padding=2)

#     # # ----------------------------------------------------
#     # # 2) LABEL CONDITIONING인 경우: 기존 방식 유지
#     # # ----------------------------------------------------
#     # else:
#     #     labels = torch.arange(1, num_classes + 1, device=device).repeat_interleave(n_per_class)

#     #     samples = sample_ddim_with_cfg(
#     #         model=unwrapped_model,
#     #         noise_scheduler=ddim_scheduler,
#     #         cond=labels,
#     #         num_inference_steps=args.eval_num_steps,
#     #         guidance_scale=1.0,
#     #         uncond_cond=None,
#     #         eta=0.0,
#     #         generator=None,
#     #     )  # (B, C, H, W)

#     #     if vae is not None:
#     #         scaling_factor = getattr(vae, "scaling_factor", 1.0)
#     #         latents = samples / scaling_factor
#     #         decoded = decode_in_chunks(vae, latents, max_batch=4)
#     #         imgs = decoded
#     #     else:
#     #         imgs = samples

#     #     imgs = imgs.clamp(-1.0, 1.0)
#     #     imgs_01 = (imgs + 1.0) * 0.5
#     #     imgs_01 = imgs_01.clamp(0.0, 1.0)

#     #     grid = make_grid(imgs_01, nrow=n_per_class, padding=2)

    
#     elif getattr(unwrapped_model, "grid_conditioning", False):
#         assert train_dataset is not None, "grid_conditioning=True 인 경우 train_dataset이 필요합니다."

#         class_counts = {c: 0 for c in class_ids}
#         ref_images = []
#         ref_grids = []
#         ref_labels = []

#         # dataset에서 (image, grid)를 class-wise로 수집
#         for idx in range(len(train_dataset)):
#             sample = train_dataset[idx]   # dict: {"image": ..., "grid": ...}

#             img  = sample["image"]        # (1, H, W)
#             grid = sample["grid"]         # (Hc, Wc) or (grid_hw, grid_hw)

#             # 가운데 셀 기준으로 label 추출 (그냥 보기 좋게 정렬용)
#             Hc, Wc = grid.shape[-2], grid.shape[-1]
#             cy, cx = Hc // 2, Wc // 2
#             lab_val = grid[cy, cx]
#             lab_int = int(lab_val)

#             if lab_int not in class_counts:
#                 continue

#             if class_counts[lab_int] < n_per_class:
#                 ref_images.append(img.unsqueeze(0))   # (1, 1, H, W)
#                 ref_grids.append(grid.unsqueeze(0))   # (1, Hc, Wc)
#                 ref_labels.append(lab_int)
#                 class_counts[lab_int] += 1

#                 if sum(class_counts.values()) == total_b:
#                     break

#         if len(ref_images) == 0:
#             raise RuntimeError("[Eval] grid conditioning용 이미지를 하나도 못 모았습니다.")

#         if len(ref_images) < total_b:
#             accelerator.print(
#                 f"[Eval] Warning: only {len(ref_images)}/{total_b} grid samples collected."
#             )

#         ref_images = torch.cat(ref_images, dim=0).to(device)   # (B, 1, H, W)
#         ref_grids  = torch.cat(ref_grids,  dim=0).to(device)   # (B, Hc, Wc)
#         B = ref_images.size(0)

#         # DDIM 세팅
#         in_channels = unet_config.get("in_channels", 1)
#         sample_size = unet_config.get("sample_size", ref_images.shape[-1])

#         ddim_scheduler.set_timesteps(args.eval_num_steps, device=device)

#         # 초기 노이즈
#         x = torch.randn(
#             (B, in_channels, sample_size, sample_size),
#             device=device,
#         )

#         # DDIM loop: grid를 condition으로 넣어줌
#         for t in tqdm(ddim_scheduler.timesteps, disable=True):
#             t_batch = torch.full((B,), t, device=device, dtype=torch.long)

#             eps = unwrapped_model(
#                 x,
#                 t_batch,
#                 grid=ref_grids,     # 🔥 여기!
#             )

#             step = ddim_scheduler.step(eps, t, x)
#             x = step.prev_sample

#         # latent → pixel
#         if vae is not None:
#             scaling_factor = getattr(vae, "scaling_factor", 1.0)
#             latents = x / scaling_factor
#             decoded = decode_in_chunks(vae, latents, max_batch=4)
#             imgs = decoded
#         else:
#             imgs = x

#         # [real | fake] 그리드 만들기 (image_conditioning branch와 동일)
#         real_imgs_01 = (ref_images.clamp(-1.0, 1.0) + 1.0) * 0.5
#         real_imgs_01 = real_imgs_01.clamp(0.0, 1.0)

#         fake_imgs_01 = (imgs.clamp(-1.0, 1.0) + 1.0) * 0.5
#         fake_imgs_01 = fake_imgs_01.clamp(0.0, 1.0)

#         pair_imgs = []
#         for i in range(B):
#             real = real_imgs_01[i]
#             fake = fake_imgs_01[i]

#             pair = torch.cat([real, fake], dim=2)  # (C, H, 2W)
#             pair = F.pad(pair, (1, 1, 1, 1), value=1.0)
#             pair_imgs.append(pair.unsqueeze(0))

#         pair_imgs = torch.cat(pair_imgs, dim=0)
#         grid = make_grid(pair_imgs, nrow=n_per_class, padding=2)


#     else:
#         # dataset에서 class-wise로 이미지 수집
#         class_counts = {c: 0 for c in class_ids}
#         ref_images = []
#         ref_labels = []

#         for idx in range(len(train_dataset)):
#             sample = train_dataset[idx]

#             if isinstance(sample, dict):
#                 # SRM sudoku: {"image": (1,H,W), "grid": (Hc,Wc), ...}
#                 img  = sample["image"]
#                 grid = sample.get("grid", None)

#                 if grid is None:
#                     continue

#                 # ⚠️ 현재는 가운데 셀만 label로 쓰는 부분 (아래 2번에서 다시 이야기)
#                 Hc, Wc = grid.shape[-2], grid.shape[-1]
#                 cy, cx = Hc // 2, Wc // 2
#                 lab_val = grid[cy, cx]
#                 lab_int = int(lab_val)
#             else:
#                 # MNIST: (img, label)
#                 img, lab = sample
#                 lab_int = int(lab)

#             if lab_int not in class_counts:
#                 continue

#             if class_counts[lab_int] < n_per_class:
#                 ref_images.append(img.unsqueeze(0))  # (1, C, H, W)
#                 ref_labels.append(lab_int)
#                 class_counts[lab_int] += 1

#                 if sum(class_counts.values()) == total_b:
#                     break

#         if len(ref_images) == 0:
#             raise RuntimeError("[Eval] label conditioning용 이미지를 하나도 못 모았습니다.")

#         if len(ref_images) < total_b:
#             accelerator.print(
#                 f"[Eval] Warning: only {len(ref_images)}/{total_b} images collected for label conditioning eval."
#             )

#         ref_images = torch.cat(ref_images, dim=0).to(device)  # (B, C, H, W)
#         labels = torch.tensor(ref_labels, device=device, dtype=torch.long)  # (B,)
#         B = ref_images.size(0)

#         # DDIM timesteps 설정
#         ddim_scheduler.set_timesteps(args.eval_num_steps, device=device)

#         # label 기반 DDIM sampling
#         samples = sample_ddim_with_cfg(
#             model=unwrapped_model,
#             noise_scheduler=ddim_scheduler,
#             cond=labels,
#             num_inference_steps=args.eval_num_steps,
#             guidance_scale=1.0,
#             uncond_cond=None,
#             eta=0.0,
#             generator=None,
#         )  # (B, C, H, W) (pixel or latent)

#         # latent → pixel (VAE 있으면 decode)
#         if vae is not None:
#             scaling_factor = getattr(vae, "scaling_factor", 1.0)
#             latents = samples / scaling_factor
#             decoded = decode_in_chunks(vae, latents, max_batch=4)
#             imgs = decoded
#         else:
#             imgs = samples

#         # real / fake 둘 다 [0,1]로 맞추기
#         real_imgs_01 = (ref_images.clamp(-1.0, 1.0) + 1.0) * 0.5
#         real_imgs_01 = real_imgs_01.clamp(0.0, 1.0)

#         fake_imgs_01 = (imgs.clamp(-1.0, 1.0) + 1.0) * 0.5
#         fake_imgs_01 = fake_imgs_01.clamp(0.0, 1.0)

#         # [real | fake] 가로 concat + 테두리
#         pair_imgs = []
#         for i in range(B):
#             real = real_imgs_01[i]  # (C,H,W)
#             fake = fake_imgs_01[i]  # (C,H,W)

#             pair = torch.cat([real, fake], dim=2)  # (C, H, 2W)
#             pair = F.pad(pair, (1, 1, 1, 1), value=1.0)  # 테두리 흰색
#             pair_imgs.append(pair.unsqueeze(0))

#         pair_imgs = torch.cat(pair_imgs, dim=0)  # (B, C, H', W')

#         # class별로 한 줄씩 → nrow = n_per_class
#         grid = make_grid(pair_imgs, nrow=n_per_class, padding=2)

#     # ----------------------------------------------------
#     # 공통: 이미지 저장
#     # ----------------------------------------------------
#     eval_dir = os.path.join(args.output_dir, "eval_samples")
#     os.makedirs(eval_dir, exist_ok=True)
#     out_path = os.path.join(eval_dir, f"step_{global_step}.png")
#     save_image(grid, out_path)

#     accelerator.print(f"[Eval] Saved eval samples to {out_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ✅ Accelerate 설정: mixed_precision + logging_dir
    logging_dir = os.path.join(args.output_dir, "logs")
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        project_config=project_config,
    )
    set_seed(args.seed + accelerator.process_index)

    if accelerator.is_main_process:
        print("Arguments:", args)

    # 트래커 (원하면 tensorboard / wandb 등)
    if accelerator.is_main_process and args.log_with is not None:
        accelerator.init_trackers(
            project_name="mnist_diffusion",
            config=vars(args),
        )

    # -----------------------
    # UNet config 로드
    # -----------------------
    with open(args.unet_config, "r") as f:
        unet_config = json.load(f)

    # ✅ 데이터/모델에서 사용할 이미지 크기 결정
    resize_image_size = args.resize_image_size         # MNIST resize (예: 32)
    pad_image_size = args.pad_image_size              # optional canvas (예: 288)
    model_image_size = pad_image_size or resize_image_size  # UNet이 보는 최종 사이즈


    if accelerator.is_main_process:
        print(f"Loaded UNet config from {args.unet_config}")
        print(f"Dataset: resize_to={resize_image_size}, pad_to={pad_image_size}")
        print(f"Model image_size (UNet sample_size) = {model_image_size}")

    # -----------------------
    # VAE 로드 (옵션)
    # -----------------------
    vae = None
    if args.vae_config is not None:
        if accelerator.is_main_process:
            print(f"Loading VAE from {args.vae_config}")
        vae = load_vae_from_config(args.vae_config, device=accelerator.device, dtype=torch.float32)


    # 🔹 VAE config에서 latent 채널 수(z_channels) + latent 해상도 읽기
    latent_in_channels = 1
    latent_spatial_size = None
    if args.vae_config is not None:
        with open(args.vae_config, "r") as f:
            vae_cfg_raw = json.load(f)

        # 보통 구조: {"target": ..., "params": { "ddconfig": {...} , ... }, "ckpt_path": ...}
        vae_params = vae_cfg_raw.get("params", {})
        vae_ddconfig = vae_params.get("ddconfig", vae_cfg_raw.get("ddconfig", None))

        if vae_ddconfig is not None:
            if "z_channels" in vae_ddconfig:
                latent_in_channels = vae_ddconfig["z_channels"]

            # 더미 forward 안 쓰고 config로 latent 해상도 계산
            # resolution = 288, ch_mult = [1,1,2,2,4] 이면
            # num_down = len(ch_mult) - 1 = 4  → 288 / 2^4 = 18
            resolution = vae_ddconfig["resolution"]
            ch_mult = vae_ddconfig["ch_mult"]
            num_down = len(ch_mult) - 1
            latent_spatial_size = resolution // (2 ** num_down)

        if accelerator.is_main_process:
            print(f"[INFO] latent_in_channels from VAE config = {latent_in_channels}")
            print(f"[INFO] latent_spatial_size from VAE config = {latent_spatial_size}")

    # 🔹 이제 UNet용 sample_size 결정
    if args.vae_config is not None and latent_spatial_size is not None:
        # VAE 쓰면 diffusion은 latent 해상도(예: 18x18) 기준으로 돌림
        unet_config["sample_size"] = latent_spatial_size
        model_image_size_for_unet = latent_spatial_size
    else:
        # VAE 안 쓰면 그냥 픽셀 해상도로
        unet_config["sample_size"] = model_image_size
        model_image_size_for_unet = model_image_size

    if accelerator.is_main_process:
        print(f"[INFO] UNet sample_size = {unet_config['sample_size']}")
        print(f"[INFO] UNet image_size  = {model_image_size_for_unet}")

    # -----------------------
    # Dataset & DataLoader
    # -----------------------
    if args.sudoku_config is None:
        # 기본: MNISTConditionalDataset
        train_dataset = MNISTConditionalDataset(
            root=args.data_dir,
            split="train",
            resize_to=resize_image_size,    # 예: 32
            pad_to=pad_image_size,          # 예: 288 -> 중앙 32x32, 나머지 0
        )
        if accelerator.is_main_process:
            print("[DATA] Using MNISTConditionalDataset")
    else:
        # SRM_dataset_cfg + SRM_conditioning_cfg를 JSON에서 읽어서 사용
        if accelerator.is_main_process:
            print(f"[DATA] Loading SRM dataset config from: {args.sudoku_config}")

        cfg = OmegaConf.load(args.sudoku_config)

        srm_ds_cfg = cfg.SRM_dataset_cfg           # name, root, image_shape, ...
        srm_cond_cfg = cfg.SRM_conditioning_cfg 
        train_dataset = get_dataset(srm_ds_cfg, srm_cond_cfg, "train")

        if accelerator.is_main_process:
            print(f"[DATA] Using SRM dataset: {srm_ds_cfg.name}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # -----------------------
    # Model & Optimizer
    # -----------------------
    # 입력 채널:
    # - 픽셀 공간: 1
    # - latent 공간: vae_config에 명시된 z_channels
    if args.vae_config is not None:
        in_channels = latent_in_channels   # 위에서 config로부터 읽어둔 값
    else:
        in_channels = 1

    # 🔹 VAE를 쓰면 UNet config의 in/out_channels도 latent에 맞게 덮어쓰기
    old_in = unet_config.get("in_channels", None)
    unet_config["in_channels"] = in_channels
    if accelerator.is_main_process:
        print("unet_config[in_channels]", unet_config["in_channels"])

    old_out = unet_config.get("out_channels", None)
    if old_out is None or old_out in (1, 3):
        unet_config["out_channels"] = in_channels

    use_grid_cond = args.sudoku_config is not None

    model = ConditionalUNet(
        num_classes=10,
        class_embed_dim=unet_config.get("cross_attention_dim", 128),
        image_size=model_image_size,          # pad/canvas 기준
        image_conditioning=args.image_conditioning,
        encoder=None,
        cond_dim=unet_config.get("cross_attention_dim", 128),
        cond_in_channels=1,
        unet_config=unet_config,
        grid_conditioning=use_grid_cond,
        grid_vocab_size=10,
        grid_hw=args.grid_hw,     
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    # -----------------------
    # diffusers DDPMScheduler (train용)
    # -----------------------
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        prediction_type="epsilon",
    )

    # -----------------------
    # Prepare with Accelerate
    # -----------------------
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    device = accelerator.device
    global_step = 0

    # 에폭/스텝 계산
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # HF 스타일 progress bar (step 기준)
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    
    if args.vae_test:
        run_vae_recon_test(
            accelerator=accelerator,
            vae=vae,
            train_dataset=train_dataset,
            args=args,
        )
        accelerator.wait_for_everyone()
        
    # -----------------------
    # Training Loop (step 기반)
    # -----------------------
    for epoch in range(1, num_epochs + 1):
        model.train()

        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break

            # ✅ MNIST vs SRM sudoku 분기
            if isinstance(batch, dict):                            
                images = batch["image"]
                grid   = batch["grid"]      # (B, Hc, Wc)
                labels = None

            else:
                # MNISTConditionalDataset: (img, label)
                images, labels = batch
                grid = None
                
            images = images.to(device)   # (B, 1, H, W)
            if labels is not None:
                labels = labels.to(device)

            batch_size = images.shape[0]

            timesteps = torch.randint(
                low=0,
                high=noise_scheduler.config.num_train_timesteps,
                size=(batch_size,),
                device=device,
            ).long()

            # -----------------------
            # pixel vs latent 분기
            # -----------------------
            if vae is not None:
                # VAE encode: 이미지 [-1,1] → latent
                with torch.no_grad():
                    img_fp32 = images.to(torch.float32)
                    latents = vae_encode(vae, img_fp32)
                    scaling_factor = getattr(vae, "scaling_factor", 1.0)
                    latents = latents * scaling_factor

                latents = latents.to(device=device, dtype=images.dtype)
                noise = torch.randn_like(latents)

                noisy_inputs = noise_scheduler.add_noise(
                    original_samples=latents,
                    noise=noise,
                    timesteps=timesteps,
                )
            else:
                # 기존 pixel-space 학습
                noise = torch.randn_like(images)
                noisy_inputs = noise_scheduler.add_noise(
                    original_samples=images,
                    noise=noise,
                    timesteps=timesteps,
                )

            # class conditioning or image conditioning
            if args.image_conditioning:
                # 🔹 이미지 기반 conditioning: encoder(cond_image) 사용
                pred_noise = model(
                    noisy_inputs,
                    timesteps,
                    cond_image=images,   # 원본(또는 padded) 이미지를 condition으로 사용
                )
            else:
                # 🔹 기존 label conditioning
                pred_noise = model(
                    noisy_inputs,
                    timesteps,
                    y=labels,
                    grid=grid,
                )

            loss = F.mse_loss(pred_noise, noise)

            loss = loss / args.grad_accum_steps
            accelerator.backward(loss)

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # 🔹 진행 상황 / loss 계산
                avg_loss = accelerator.gather(loss.detach()).mean().item() * args.grad_accum_steps

                # 🔹 tqdm progress bar 업데이트
                progress_bar.update(1)
                logs = {
                    "step_loss": f"{avg_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
                progress_bar.set_postfix(**logs)

                # 🔹 accelerator tracker 로깅
                accelerator.log(
                    {"train/step_loss": avg_loss, "train/lr": optimizer.param_groups[0]["lr"]},
                    step=global_step,
                )

                # 콘솔 로그 (간격 널널하게)
                if accelerator.is_main_process and (global_step % args.log_every == 0):
                    print(
                        f"Step [{global_step}/{args.max_train_steps}] "
                        f"Epoch [{epoch}/{num_epochs}] "
                        f"Loss: {avg_loss:.4f}"
                    )

                # evaluation
                if (global_step % args.eval_every == 0):
                    run_evaluation(
                        accelerator=accelerator,
                        model=model,
                        noise_scheduler=noise_scheduler,
                        unet_config=unet_config,
                        args=args,
                        global_step=global_step,
                        vae=vae,
                        train_dataset=train_dataset,
                    )

                # save checkpoint
                if (global_step % args.save_every == 0) and accelerator.is_main_process:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pt")
                    accelerator.print(f"Saving checkpoint to {ckpt_path}")
                    unwrapped_model = accelerator.unwrap_model(model)
                    state = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                        "unet_config": unet_config,
                    }
                    torch.save(state, ckpt_path)

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_ckpt = os.path.join(args.output_dir, f"checkpoint_final.pt")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "unet_config": unet_config,
            },
            final_ckpt,
        )
        print(f"Training finished at step {global_step}. Final checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()
