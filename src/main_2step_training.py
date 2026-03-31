# train_mnist_diffusion.py

import argparse
import os
import sys
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
from SRM.evaluation.sudoku_eval_only import MnistSudokuEvalOnly


from SRM.type_extensions import ConditioningCfg
from omegaconf import OmegaConf

from model import ConditionalUNet
from sampling import sample_ddim_with_cfg

from concurrent.futures import ThreadPoolExecutor
import importlib

def parse_step_from_dir(path: str) -> int:
    # accepts ".../step12345" or ".../step12345/"
    base = os.path.basename(os.path.normpath(path))
    if base.startswith("step"):
        try:
            return int(base.replace("step", ""))
        except Exception:
            pass
    # fallback: search "step" in full path
    if "step" in path:
        try:
            return int(path.split("step")[-1].split(os.sep)[0])
        except Exception:
            pass
    return 0


def compute_image_token_hw(
    *,
    image_h: int,
    image_w: int,
    downsample_factor: int,
) -> tuple[int, int, int]:
    """
    ImageCondition2DEncoder는 downsample_factor=2^k 만큼 stride-2 down을 k번 하므로
    출력 토큰 해상도는 대략 (H/down, W/down).
    - 현재 모델 코드가 h = Hc // downsample_factor 로 가정하고 있으니 동일하게 맞춤.
    """
    h = image_h // downsample_factor
    w = image_w // downsample_factor
    L = h * w
    return h, w, L


def compute_grid_token_hw(*, grid_hw: int) -> tuple[int, int, int]:
    h = grid_hw
    w = grid_hw
    L = h * w
    return h, w, L

def count_params(module: torch.nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def format_n(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n/1_000:.3f}K"
    return str(n)

def print_model_sizes(
    accelerator: Accelerator,
    model: torch.nn.Module,
    *,
    model_image_size: int,
    image_conditioning: bool,
    grid_conditioning: bool,
    grid_hw: int,
    concat_downsample_factor: int,
    patch_conditioning: bool = False,
    patch_grid_size: int = 9,
):
    if not accelerator.is_main_process:
        return

    m = accelerator.unwrap_model(model)

    enc = getattr(m, "encoder", None)
    unet = getattr(m, "unet", None)

    total_all, train_all = count_params(m)
    accelerator.print(f"[MODEL] total: {format_n(total_all)} params (trainable {format_n(train_all)})")

    if enc is not None:
        total_enc, train_enc = count_params(enc)
        accelerator.print(f"[MODEL] encoder: {format_n(total_enc)} params (trainable {format_n(train_enc)})")
    else:
        accelerator.print("[MODEL] encoder: None")

    if unet is not None:
        total_unet, train_unet = count_params(unet)
        accelerator.print(f"[MODEL] unet: {format_n(total_unet)} params (trainable {format_n(train_unet)})")
    else:
        accelerator.print("[MODEL] unet: None")

    # -----------------------------
    # ✅ conditioning token stats (forward 없이)
    # -----------------------------
    if image_conditioning:
        if patch_conditioning:
            h, w, L = patch_grid_size, patch_grid_size, patch_grid_size * patch_grid_size
            accelerator.print(
                f"[COND] image_conditioning (patchwise) tokens: h={h}, w={w}, L={L} (grid={patch_grid_size}x{patch_grid_size})"
            )
        else:
            h, w, L = compute_image_token_hw(
                image_h=model_image_size,
                image_w=model_image_size,
                downsample_factor=concat_downsample_factor,
            )
            accelerator.print(
                f"[COND] image_conditioning tokens: h={h}, w={w}, L={L} "
                f"(image={model_image_size}x{model_image_size}, down={concat_downsample_factor})"
            )

    elif grid_conditioning:
        h, w, L = compute_grid_token_hw(grid_hw=grid_hw)
        accelerator.print(
            f"[COND] grid_conditioning tokens: h={h}, w={w}, L={L} (grid_hw={grid_hw})"
        )

    else:
        accelerator.print("[COND] class_conditioning tokens: L=1")

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

    # resume/save with accelerate state
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Path to an accelerate checkpoint directory (e.g. outputs/.../ckpt/step15000).",
    )

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
        "--classifier_pth",
        type=str,
        default=None,
        help="MNIST Classifier path"
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

    parser.add_argument("--concat_conditioning", action="store_true")
    parser.add_argument("--concat_downsample_factor", type=int, default=16)
    parser.add_argument("--concat_channels", type=int, default=4)
    parser.add_argument("--cond_dim", type=int, default=None)
    parser.add_argument("--use_fsq", action="store_true")
    parser.add_argument("--fsq_levels", type=int, nargs="+",
        default=[8, 8, 8, 5],
        help="FSQ quantization levels per codebook, e.g. --fsq_levels 8 8 8 5",
    )
    parser.add_argument("--fsq_drop_quant_p", type=float, default=0.0)
    parser.add_argument("--fsq_corrupt_tokens_p", type=float, default=0.0)
    # ---------------------------------------
    parser.add_argument("--use_vq_discretizer", action="store_true",)
    parser.add_argument("--vq_loss_weight", type=float, default=0.1,)
    parser.add_argument("--vq_codebook_size", type=int, default=9,
        help="VQ codebook size (for VQ discretizer).",
    )
    parser.add_argument("--vq_beta", type=float, default=0.25,
        help="VQ commitment loss coefficient.",
    )
        # --- patchwise image conditioning encoder ---
    parser.add_argument(
        "--patch_conditioning",
        action="store_true",
        help="If set (and --image_conditioning), use patchwise encoder that splits cond_image into grid×grid patches.",
    )
    parser.add_argument(
        "--patch_grid_size",
        type=int,
        default=9,
        help="Patch grid size for patchwise encoder (e.g., 9 means 9x9 patches).",
    )
    parser.add_argument(
        "--sudoku_eval_grid_size",
        type=int,
        default=9,
        help="Grid size for Sudoku evaluator (N means NxN). Use 1 to disable Sudoku rule check and only discretize.",
    )

    # diffusion (for DDPMScheduler)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_schedule", type=str, default="linear",
                        choices=["linear", "scaled_linear", "squaredcos_cap_v2"])
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample", "v_prediction"],
        help="What the UNet predicts: epsilon, x_0(sample), or v_prediction.",
    )
    parser.add_argument("--uncond_drop_prob", type=float, default=0.1)
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                    help="CFG scale for eval sampling. 1.0 means no CFG.")


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
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    dataset: MNISTConditionalDataset | None = None,
    in_channels: int = 1,
    guidance_scale: float = 1.0,
    filename_suffix: str = "",
    sudoku_evaluator=None,
):

    accelerator.print(f"[Eval] Running evaluation at step {global_step}...")

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    device = accelerator.device

    # 어떤 conditioning 모드인지 확인
    use_image_cond = getattr(unwrapped_model, "image_conditioning", False)
    use_grid_cond  = getattr(unwrapped_model, "grid_conditioning", False)

    if use_image_cond or use_grid_cond:
        assert dataset is not None, "image/grid conditioning에서는 train_dataset이 필요합니다."

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

    for idx in range(len(dataset)):
        sample = dataset[idx]

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
        if grid is not None:
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
    
    ref_grids = torch.cat(ref_grids, dim=0).to(device)  # (B, Hc, Wc)

    if not use_image_cond and not use_grid_cond:
        labels = torch.tensor(ref_labels, device=device, dtype=torch.long)  # (B,)
    else:
        labels = None

    # --- eval diffusion state shape 결정 (train과 동일하게) ---
    sample_size = unet_config["sample_size"] 
    ddim_scheduler.set_timesteps(args.eval_num_steps, device=device)
    x = torch.randn(
        (B, in_channels, sample_size, sample_size),
        device=device,
    )
    
    cond_tokens, tok_ids = unwrapped_model.cond_encoding(
        y=labels,
        cond_image=ref_images,
        grid=ref_grids,
        return_token_ids=True,
    )

    if guidance_scale != 1.0:
        uncond_tokens = unwrapped_model.cond_encoding(
            y=labels,              # shape 맞추려고 그냥 같이 넣어도 되고
            cond_image=ref_images, # image/grid 모드면 L 맞추기 위해 필요
            grid=ref_grids,
            return_uncond=True,     # ✅ null로 강제
        )

    for step_idx, t in enumerate(tqdm(ddim_scheduler.timesteps, disable=True)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        if guidance_scale == 1.0:
            eps = unwrapped_model(x, t_batch, encoder_hidden_states=cond_tokens, cond_image=ref_images,)
        else:
            eps_c = unwrapped_model(x, t_batch, encoder_hidden_states=cond_tokens, cond_image=ref_images,)
            eps_u = unwrapped_model(x, t_batch, encoder_hidden_states=uncond_tokens, cond_image=ref_images,)
            eps = eps_u + guidance_scale * (eps_c - eps_u)

        # 두 번째 step에서 바로 x0 예측으로 나가기
        if step_idx == 1:
            x = ddim_scheduler.step(eps, t, x).pred_original_sample
            break
        else:
            x = ddim_scheduler.step(eps, t, x).prev_sample

    # VAE decode (있으면 latent → pixel)
    if vae is not None:
        scaling_factor = getattr(vae, "scaling_factor", 1.0)
        latents = x / scaling_factor
        imgs = decode_in_chunks(vae, latents, max_batch=4)
    else:
        imgs = x

    if not accelerator.is_main_process:
        unwrapped_model.train()
        return

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
    suf = f"_{filename_suffix}" if filename_suffix else ""
    out_path = os.path.join(eval_dir, f"step_{global_step}{suf}.png")
    save_image(grid, out_path)
    
    
    # ---------- 6) Sudoku evaluator (optional) ----------
    if sudoku_evaluator is not None:
        # evaluator expects [-1, 1]
        fake_imgs_m11 = fake_imgs_01 * 2.0 - 1.0

        # (B,) bool labels, (B,) distance, () acc, (B,9,9) discrete
        s_eval = sudoku_evaluator.eval_images(fake_imgs_m11)

        # ---- GT grid와 비교 (가능할 때만) ----
        # ref_grids: (B, Hc, Wc) 라고 가정 (SRM sudoku)
        # grid가 "숫자 그리드(9x9)" 형태면 보통 (B,9,9)이거나 중앙 crop 등이 필요할 수 있음.
        # 아래는 (B,9,9)라고 가정하고 비교. 아니라면 변환해서 넣어줘야 함.
        wrong_mask = None
        n_wrong = None
        cell_acc = None
        acc_tok_vs_gt = None
        acc_tok_vs_pred = None

        if ref_grids is not None:
            gt = ref_grids
            # gt shape 정리: (B,9,9)로 맞춘다고 가정
            # 만약 gt가 (B,Hc,Wc)이고 그 안에 9x9가 어딘가에 있다면, 그 부분을 잘라서 사용하세요.
            if gt.dim() == 3 and gt.shape[1] != s_eval["discrete"].shape[1]:
                # 예: 중앙 9x9 crop (필요시 여기 로직을 프로젝트 grid 포맷에 맞게 수정)
                Gh, Gw = s_eval["discrete"].shape[1], s_eval["discrete"].shape[2]
                Hc, Wc = gt.shape[1], gt.shape[2]
                sy = (Hc - Gh) // 2
                sx = (Wc - Gw) // 2
                gt = gt[:, sy:sy+Gh, sx:sx+Gw]

            gt = gt.to(s_eval["discrete"].device).long()

            pred_grid = s_eval["discrete"].long()
            wrong_mask = (pred_grid != gt)  # (B,9,9) bool
            n_wrong = wrong_mask.flatten(1).sum(dim=1)  # (B,)
            cell_acc = (~wrong_mask).float().mean()     # scalar

            # ✅ 여기 추가 (전체 batch wrong < 10일 때만 출력)
            total_wrong = int(wrong_mask.sum().item())
            if total_wrong < 10:
                accelerator.print(f"[Eval][Diff] TOTAL wrong cells={total_wrong} (<10)")

                wm = wrong_mask.detach().cpu()
                gt_cpu = gt.detach().cpu()
                pr_cpu = pred_grid.detach().cpu()

                for bi in range(wm.shape[0]):
                    coords = torch.nonzero(wm[bi], as_tuple=False)
                    if coords.numel() == 0:
                        continue
                    accelerator.print(f"[Eval][Diff] sample#{bi} wrong={coords.shape[0]}")
                    for r, c in coords.tolist():
                        accelerator.print(
                            f"  - (row={r}, col={c}) GT={int(gt_cpu[bi,r,c])} PRED={int(pr_cpu[bi,r,c])}"
                        )
                        
        # ---- 추가: GT image에서 evaluator가 뽑은 grid와 비교 ----
        real_imgs_m11 = real_imgs_01 * 2.0 - 1.0
        s_gtimg = sudoku_evaluator.eval_images(real_imgs_m11)
        gt_from_img = s_gtimg["discrete"].to(pred_grid.device).long()

        wrong_mask_img = (pred_grid != gt_from_img)   # (B,9,9)
        n_wrong_img = wrong_mask_img.flatten(1).sum(dim=1)  # (B,)
        cell_acc_img = (~wrong_mask_img).float().mean()     # scalar

        total_wrong_img = int(wrong_mask_img.sum().item())
        if total_wrong_img < 100:
            accelerator.print(f"[Eval][DiffImg] TOTAL wrong cells={total_wrong_img} (<10)")

            wm = wrong_mask_img.detach().cpu()
            gt_cpu = gt_from_img.detach().cpu()
            pr_cpu = pred_grid.detach().cpu()

            for bi in range(wm.shape[0]):
                coords = torch.nonzero(wm[bi], as_tuple=False)
                if coords.numel() == 0:
                    continue
                accelerator.print(f"[Eval][DiffImg] sample#{bi} wrong={coords.shape[0]}")
                for r, c in coords.tolist():
                    accelerator.print(
                        f"  - (row={r}, col={c}) GTimg={int(gt_cpu[bi,r,c])} PRED={int(pr_cpu[bi,r,c])}"
                    )

        if args.use_fsq or args.use_vq_discretizer:
            # ============================================================
            # ====== 추가 분석: 토큰ID -> 숫자 매핑 및 시각화 ======
            # ============================================================
            ##############################################################
            # ====== 추가 import 필요 ======
            from PIL import Image, ImageDraw, ImageFont
            import math

            # ============================================================
            # [TokID -> Digit] mapping + visualize (GT / tok2digit / pred_grid)
            # ============================================================

            # 0) tok_ids 준비 (이미 위에서 뽑았다고 했으니 여기선 shape만 보정/확인)
            tok_ids_2d = tok_ids.view(B, 9, 9).long()  # (B,9,9)

            # gt/pred_grid는 네 코드에서 이미 여기서 확정됨:
            # gt = ... (B,9,9)
            # pred_grid = s_eval["discrete"].long()  # (B,9,9)
            gt_9 = gt.long()
            pred_9 = pred_grid.long()

            assert tok_ids_2d.shape == gt_9.shape, f"tok_ids={tok_ids_2d.shape}, gt={gt_9.shape}"
            assert pred_9.shape == gt_9.shape, f"pred={pred_9.shape}, gt={gt_9.shape}"

            # 1) GT vs pred_grid mismatch mask (빨간 테두리용)
            wrong_mask_pred = (pred_9 != gt_9)  # (B,9,9) bool

            # 2) tok_id -> digit 통계 매핑 만들기 (batch 전체에서)
            #    vocab은 tok_ids에서 자동 추정
            vocab = int(tok_ids_2d.max().item()) + 1  # token book 개수
            tid_flat = tok_ids_2d.reshape(-1)         # (B*81,)
            gt_flat  = gt_9.reshape(-1).clamp(0, 9)   # digit 0~9 가정

            # counts[tok, digit] 만들기 (vectorized)
            # idx = tok*10 + digit
            idx = (tid_flat * 10 + gt_flat).to(torch.long)
            counts = torch.bincount(idx, minlength=vocab * 10).view(vocab, 10)  # (vocab,10)

            tok2digit = counts.argmax(dim=1)  # (vocab,)
            tok_conf  = counts.max(dim=1).values.float() / (counts.sum(dim=1).float() + 1e-9)

            # 3) tok_id grid를 digit grid로 변환
            pred_from_tok = tok2digit[tok_ids_2d]  # (B,9,9)

            # 4) 비교 지표(원하면 로그)
            wrong_tok_vs_gt = (pred_from_tok != gt_9)
            acc_tok_vs_gt = (~wrong_tok_vs_gt).float().mean().item()

            wrong_tok_vs_pred = (pred_from_tok != pred_9)
            acc_tok_vs_pred = (~wrong_tok_vs_pred).float().mean().item()

            accelerator.print(
                f"[Eval][Tok2Digit] vocab={vocab} "
                f"acc(tok->digit vs GT)={acc_tok_vs_gt:.4f} "
                f"acc(tok->digit vs PRED)={acc_tok_vs_pred:.4f}"
            )

            # 5) digit confusion(뭐랑 헷갈렸는지) - GT vs pred_grid, GT vs tok2digit
            def confusion_10x10(gt_grid, pr_grid):
                g = gt_grid.reshape(-1).clamp(0, 9).to(torch.long)
                p = pr_grid.reshape(-1).clamp(0, 9).to(torch.long)
                cm = torch.bincount(g * 10 + p, minlength=100).view(10, 10)  # rows=GT, cols=PRED
                return cm

            cm_pred = confusion_10x10(gt_9, pred_9)
            cm_tok  = confusion_10x10(gt_9, pred_from_tok)

            def print_full_confusion(cm, name, accelerator, digits=10):
                # cm: (10,10) rows=GT, cols=PRED
                if not accelerator.is_main_process:
                    return
                accelerator.print(f"[Eval][ConfusionFull] {name} (rows=GT, cols=Pred)")
                for gt_d in range(digits):
                    row = cm[gt_d]
                    total = int(row.sum().item())
                    if total == 0:
                        continue
                    parts = []
                    for pr_d in range(digits):
                        cnt = int(row[pr_d].item())
                        if cnt == 0:
                            continue
                        parts.append(f"{pr_d}:{cnt}")
                    accelerator.print(f"  GT {gt_d} (n={total}) -> " + ", ".join(parts))

            # 전체 출력
            print_full_confusion(cm_pred, "GT->PRED", accelerator)
            print_full_confusion(cm_tok,  "GT->TOK2DIG", accelerator)

            # 6) ====== 숫자 그리드 렌더링 (PIL) ======
            def render_digit_grid(
                grid_9x9,
                wrong_mask_9x9=None,     # 빨간 테두리 (GT vs PRED mismatch)
                bg_mask_9x9=None,        # 연분홍 배경 (GT vs TOK2DIG mismatch)
                cell=34, pad=3, border=3, title=None,
                font_size=18,
            ):
                """
                grid_9x9: (9,9) tensor/ndarray of ints
                wrong_mask_9x9: (9,9) bool -> True면 빨간 테두리
                bg_mask_9x9: (9,9) bool -> True면 연분홍 배경
                """
                grid_9x9 = grid_9x9.detach().cpu().numpy()
                if wrong_mask_9x9 is not None:
                    wrong_mask_9x9 = wrong_mask_9x9.detach().cpu().numpy()
                if bg_mask_9x9 is not None:
                    bg_mask_9x9 = bg_mask_9x9.detach().cpu().numpy()

                # title을 더 타이트하게
                title_h = 18 if title else 0
                W = 9 * cell + 2 * pad
                H = 9 * cell + 2 * pad + title_h

                img = Image.new("RGB", (W, H), (255, 255, 255))
                draw = ImageDraw.Draw(img)

                # 폰트: 있으면 truetype로 크게, 없으면 default
                font = None
                try:
                    # 컨테이너에 보통 있는 폰트 경로(없으면 except)
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()

                if title:
                    draw.text((pad, 0), title, fill=(0, 0, 0), font=font)

                y0 = title_h

                pink = (255, 230, 235)  # 연분홍 배경

                for r in range(9):
                    for c in range(9):
                        x1 = pad + c * cell
                        y1 = y0 + pad + r * cell
                        x2 = x1 + cell
                        y2 = y1 + cell

                        # 배경 채우기 (GT vs TOK2DIG mismatch)
                        if bg_mask_9x9 is not None and bool(bg_mask_9x9[r, c]):
                            draw.rectangle([x1, y1, x2, y2], fill=pink)

                        # 기본 셀 테두리 (연한 회색)
                        draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200), width=1)

                        val = int(grid_9x9[r, c])
                        s = str(val)

                        # text centering (Pillow>=10)
                        bbox = draw.textbbox((0, 0), s, font=font)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                        tx = x1 + (cell - tw) / 2
                        ty = y1 + (cell - th) / 2
                        draw.text((tx, ty), s, fill=(0, 0, 0), font=font)

                        # 빨간 테두리 (GT vs PRED mismatch)
                        if wrong_mask_9x9 is not None and bool(wrong_mask_9x9[r, c]):
                            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=border)

                return img
            def tile_images(img_list, nrow, pad=10, bg=(255,255,255)):
                """
                img_list: list of PIL Images (same size)
                nrow: columns
                """
                if len(img_list) == 0:
                    return None
                w, h = img_list[0].size
                ncol = nrow
                nrows = math.ceil(len(img_list) / ncol)
                out_w = ncol * w + (ncol + 1) * pad
                out_h = nrows * h + (nrows + 1) * pad
                canvas = Image.new("RGB", (out_w, out_h), bg)

                for i, im in enumerate(img_list):
                    rr = i // ncol
                    cc = i % ncol
                    x = pad + cc * (w + pad)
                    y = pad + rr * (h + pad)
                    canvas.paste(im, (x, y))
                return canvas

            # 7) batch 전체를 타일로 모아서 GT / TOK2DIG / PRED 3장 저장
            #    빨간 테두리는 "GT vs PRED 불일치" 기준을 모든 그림에 동일하게 적용
            gt_imgs = []
            tok_imgs = []
            pred_imgs = []

            # GT vs TOK2DIG mismatch (분홍 배경)
            wrong_mask_tok = (pred_from_tok != gt_9)  # (B,9,9)

            for i in range(B):
                wm_red = wrong_mask_pred[i]   # GT vs PRED mismatch -> 빨간 테두리
                wm_pink = wrong_mask_tok[i]   # GT vs TOK2DIG mismatch -> 분홍 배경

                gt_imgs.append(render_digit_grid(
                    gt_9[i],
                    wrong_mask_9x9=wm_red,
                    bg_mask_9x9=wm_pink,
                    title=None #f"GT #{i}"
                ))
                tok_imgs.append(render_digit_grid(
                    pred_from_tok[i],
                    wrong_mask_9x9=wm_red,
                    bg_mask_9x9=wm_pink,
                    title=None #f"TOK2DIG #{i}"
                ))
                pred_imgs.append(render_digit_grid(
                    pred_9[i],
                    wrong_mask_9x9=wm_red,
                    bg_mask_9x9=wm_pink,
                    title=None #f"PRED #{i}"
                ))

            # 타일 nrow는 기존 n_per_class 쓰면 보기 좋음
            nrow = n_per_class if 'n_per_class' in locals() else min(B, 8)

            gt_canvas  = tile_images(gt_imgs,  nrow=nrow)
            tok_canvas = tile_images(tok_imgs, nrow=nrow)
            pr_canvas  = tile_images(pred_imgs, nrow=nrow)

            viz_dir = os.path.join(eval_dir, "grid_digits")
            os.makedirs(viz_dir, exist_ok=True)

            gt_path  = os.path.join(viz_dir, f"step_{global_step}{suf}_GT.png")
            tok_path = os.path.join(viz_dir, f"step_{global_step}{suf}_TOK2DIG.png")
            pr_path  = os.path.join(viz_dir, f"step_{global_step}{suf}_PRED.png")

            gt_canvas.save(gt_path)
            tok_canvas.save(tok_path)
            pr_canvas.save(pr_path)

            accelerator.print(f"[Eval] Saved digit grids:\n  {gt_path}\n  {tok_path}\n  {pr_path}")
            # ============================================================

            # 필요하면 result를 밖에서 쓰기 좋게 묶어서 리턴/저장
            # 예) 디버깅용으로 첫 샘플 비교 출력
            # accelerator.print("GT[0]:\n", gt[0])
            # accelerator.print("PR[0]:\n", s_eval["discrete"][0])
            # accelerator.print("WRONG[0]:\n", wrong_mask[0].int())
        

        # ---- 로그/출력용 dict로 합치기 ----
        # (wandb나 tensorboard에 올리고 싶으면 여기서 accelerator.log로 넘기면 됨)
        accelerator.print(
            f"[Eval][Sudoku] step={global_step} "
            f"rule_acc={float(s_eval['accuracy']):.4f} "
            f"dist_mean={float(s_eval['distance'].float().mean()):.2f} "
            + (f" cell_acc_grid={float(cell_acc):.4f} wrong_mean_grid={float(n_wrong.float().mean()):.2f}"
                if (cell_acc is not None) else "")
            + (f" cell_acc_img={float(cell_acc_img):.4f} wrong_mean_img={float(n_wrong_img.float().mean()):.2f}"
                if (cell_acc_img is not None) else "")
        )

        if accelerator.is_main_process:
            log_dict = {
                "eval/sudoku_rule_acc": s_eval["accuracy"].item(),
                "eval/sudoku_dist_mean": s_eval["distance"].float().mean().item(),
            }

            # GT 비교 가능할 때만
            if cell_acc is not None:
                log_dict.update({
                    "eval/sudoku_cell_acc": cell_acc.item(),
                    "eval/sudoku_wrong_mean": n_wrong.float().mean().item(),
                })
            log_dict.update({
                "eval/sudoku_cell_acc_img": cell_acc_img.item(),
                "eval/sudoku_wrong_mean_img": n_wrong_img.float().mean().item(),
            })

            if acc_tok_vs_gt is not None:
                log_dict.update({
                    "eval/tok2digit_acc_vs_gt": acc_tok_vs_gt,
                    "eval/tok2digit_acc_vs_pred": acc_tok_vs_pred,
                })
                                
            accelerator.log(log_dict, step=global_step)  # ✅ 이거 추가

            
    unwrapped_model.train()

    accelerator.print(f"[Eval] Saved eval samples to {out_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    
    # 🔹 CLI 인자 & 실행 커맨드 기록
    args_path = os.path.join(args.output_dir, "run_config.json")
    with open(args_path, "w") as f:
        json.dump(
            {
                "cmd": " ".join(sys.argv),
                "args": vars(args),
            },
            f,
            indent=2,
            sort_keys=True,
        )
        

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
        config = vars(args).copy()
        config["fsq_levels"] = ",".join(map(str, args.fsq_levels)) 
        accelerator.init_trackers(
            project_name="mnist_diffusion",
            config=config,
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

        sudoku_evaluator = None

    else:
        # SRM_dataset_cfg + SRM_conditioning_cfg를 JSON에서 읽어서 사용
        if accelerator.is_main_process:
            print(f"[DATA] Loading SRM dataset config from: {args.sudoku_config}")

        cfg = OmegaConf.load(args.sudoku_config)

        srm_ds_cfg = cfg.SRM_dataset_cfg           # name, root, image_shape, ...
        srm_cond_cfg = cfg.SRM_conditioning_cfg 
        train_dataset = get_dataset(srm_ds_cfg, srm_cond_cfg, "train")
        validation_dataset = get_dataset(srm_ds_cfg, srm_cond_cfg, "val")

        if accelerator.is_main_process:
            print(f"[DATA] Using SRM dataset: {srm_ds_cfg.name}")


        sudoku_evaluator = MnistSudokuEvalOnly(mnist_classifier_path=args.classifier_pth, grid_size= (args.sudoku_eval_grid_size,args.sudoku_eval_grid_size))
        # sudoku_evaluator = None

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

    use_image_cond = bool(args.image_conditioning)
    use_grid_cond  = (args.sudoku_config is not None) and (not use_image_cond)

    model = ConditionalUNet(
        num_classes=10,
        class_embed_dim=unet_config.get("cross_attention_dim", 128),
        image_size=model_image_size,          # pad/canvas 기준
        encoder=None,
        cond_dim=args.cond_dim,
        cond_in_channels=1,
        unet_config=unet_config,
        grid_conditioning=use_grid_cond,
        image_conditioning=use_image_cond,
        grid_vocab_size=10,
        grid_hw=args.grid_hw,     
        uncond_drop_prob=args.uncond_drop_prob,
        concat_conditioning=args.concat_conditioning,
        concat_downsample_factor=args.concat_downsample_factor,
        concat_channels=args.concat_channels,
        use_fsq=args.use_fsq,
        fsq_levels=args.fsq_levels,
        fsq_drop_quant_p=args.fsq_drop_quant_p,
        fsq_corrupt_tokens_p=args.fsq_corrupt_tokens_p,
        use_vq_discretizer=args.use_vq_discretizer,
        vq_codebook_size=args.vq_codebook_size,
        vq_beta=args.vq_beta,
        patch_conditioning=args.patch_conditioning,
        patch_grid_size=args.patch_grid_size,
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
        prediction_type=args.prediction_type,
    )

    # -----------------------
    # Prepare with Accelerate
    # -----------------------
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    
    loaded_step = 0
    if args.resume_dir is not None:
        if accelerator.is_main_process:
            accelerator.print(f"[RESUME] Loading accelerate state from: {args.resume_dir}")

        # restore model/optimizer/etc
        accelerator.load_state(args.resume_dir)

        # infer step from folder name (DiffusionTrainer 스타일)
        loaded_step = parse_step_from_dir(args.resume_dir)

        if accelerator.is_main_process:
            accelerator.print(f"[RESUME] loaded_step(parsed) = {loaded_step}")


    device = accelerator.device
    global_step = loaded_step

    # accelerator.wait_for_everyone() 
    
    print_model_sizes(
    accelerator,
    model,
    model_image_size=model_image_size,              # pixel-space cond_image 크기 (pad/resize 반영)
    image_conditioning=use_image_cond,
    grid_conditioning=use_grid_cond,
    grid_hw=args.grid_hw,
    concat_downsample_factor=args.concat_downsample_factor,
    patch_conditioning=args.patch_conditioning,
    patch_grid_size=args.patch_grid_size,
    )

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
        return

    # -----------------------
    # Training Loop (step 기반)
    # -----------------------
    model.train()
    epoch = 0

    while global_step < args.max_train_steps:
        epoch += 1
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
            # if accelerator.is_main_process:
            #     accelerator.print(
            #         f"[IMG] images: shape={tuple(images.shape)} dtype={images.dtype} device={images.device} "
            #         f"min={images.min().item():.6f} max={images.max().item():.6f} "
            #         f"mean={images.mean().item():.6f} std={images.std().item():.6f}"
            #     )
            #     if labels is not None:
            #         accelerator.print(
            #             f"[LBL] labels: shape={tuple(labels.shape)} dtype={labels.dtype} "
            #             f"min={labels.min().item()} max={labels.max().item()}"
            #         )
            #     if grid is not None:
            #         accelerator.print(
            #             f"[GRID] grid: shape={tuple(grid.shape)} dtype={grid.dtype} device={grid.device} "
            #             f"min={grid.min().item()} max={grid.max().item()}"
            #         )
                
            if labels is not None:
                labels = labels.to(device)

            batch_size = images.shape[0]

            timesteps = torch.randint(
                low=960,
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
                    x0 = vae_encode(vae, img_fp32)
                    scaling_factor = getattr(vae, "scaling_factor", 1.0)
                    x0 = x0 * scaling_factor

                x0 = x0.to(device=device, dtype=images.dtype)
                noise = torch.randn_like(x0)

                noisy_inputs = noise_scheduler.add_noise(
                    original_samples=x0,
                    noise=noise,
                    timesteps=timesteps,
                )
            else:
                x0 = images
                # 기존 pixel-space 학습
                noise = torch.randn_like(x0)
                noisy_inputs = noise_scheduler.add_noise(
                    original_samples=x0,
                    noise=noise,
                    timesteps=timesteps,
                )

            vq_loss = None
            # class conditioning or image conditioning
            if args.image_conditioning:
                if not args.use_vq_discretizer:
                    # 🔹 이미지 기반 conditioning: encoder(cond_image) 사용
                    pred_noise = model(
                        noisy_inputs,
                        timesteps,
                        cond_image=images,   # 원본(또는 padded) 이미지를 condition으로 사용
                    )
                else:
                    # 🔹 VQ discretizer 기반 이미지 conditioning
                    pred_noise, vq_loss = model(
                        noisy_inputs,
                        timesteps,
                        cond_image=images,   # 원본(또는 padded) 이미지를 condition으로 사용
                        return_aux_loss=True,
                    )
            else:
                # 🔹 기존 label conditioning
                pred_noise = model(
                    noisy_inputs,
                    timesteps,
                    y=labels,
                    grid=grid,
                )


            if args.prediction_type == "epsilon":
                target = noise
            elif args.prediction_type == "sample":          # x_0 prediction
                target = x0
            elif args.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    original_samples=x0,
                    noise=noise,
                    timesteps=timesteps,
                )
            else:
                raise ValueError(f"Unknown prediction_type: {args.prediction_type}")
            
            loss = F.mse_loss(pred_noise, target)

            if vq_loss is not None:
                loss = loss + args.vq_loss_weight * vq_loss

            loss = loss / args.grad_accum_steps
            accelerator.backward(loss)

            if (step + 1) % args.grad_accum_steps == 0:
                if accelerator.is_main_process and (global_step % args.log_every == 0):
                    m = accelerator.unwrap_model(model)
                    if getattr(m, "null_cond", None) is not None:
                        null_before = m.null_cond.detach().float().clone()

                optimizer.step()

                # after optimizer.step()
                if accelerator.is_main_process and (global_step % args.log_every == 0):
                    m = accelerator.unwrap_model(model)
                    if getattr(m, "null_cond", None) is not None:
                        delta = (m.null_cond.detach().float() - null_before).abs()
                        accelerator.print(
                            f"[UPD] null_cond |Δ| mean={delta.mean().item():.6e} max={delta.max().item():.6e}"
                        )
        
                optimizer.zero_grad()
                global_step += 1

                # 🔹 진행 상황 / loss 계산
                avg_loss = accelerator.gather(loss.detach()).mean().item() * args.grad_accum_steps

                # 🔹 tqdm progress bar 업데이트
                progress_bar.update(1)
                logs = {
                    "step_loss": f"{avg_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "vq_loss": f"{(args.vq_loss_weight * vq_loss).item() if vq_loss is not None else 0:.4f}",
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

                    # CFG (g=args.guidance_scale) - 스케일이 1이면 굳이 또 안 돌려도 됨
                    if args.guidance_scale != 1.0:
                        run_evaluation(
                            accelerator=accelerator,
                            model=model,
                            noise_scheduler=noise_scheduler,
                            unet_config=unet_config,
                            args=args,
                            global_step=global_step,
                            vae=vae,
                            dataset=validation_dataset,
                            in_channels=in_channels,
                            guidance_scale= args.guidance_scale,
                            filename_suffix=f"cfg{args.guidance_scale:g}",
                            sudoku_evaluator=sudoku_evaluator,
                        )
                        
                    run_evaluation(
                        accelerator=accelerator,
                        model=model,
                        noise_scheduler=noise_scheduler,
                        unet_config=unet_config,
                        args=args,
                        global_step=global_step,
                        vae=vae,
                        dataset=validation_dataset,
                        in_channels=in_channels,
                        guidance_scale= 1.0,
                        sudoku_evaluator=sudoku_evaluator,
                    )

                        
                # save checkpoint (accelerate state)
                if (global_step % args.save_every == 0):
                    ckpt_root = os.path.join(args.output_dir, "ckpt")
                    ckpt_path = os.path.join(ckpt_root, f"step{global_step}")
                    accelerator.print(f"[CKPT] Saving accelerate state to {ckpt_path}")
                    accelerator.save_state(ckpt_path)

                    # (옵션) run_config / unet_config 같은 메타도 같이 저장하고 싶으면 main process에서만 별도로 저장
                    if accelerator.is_main_process:
                        meta = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "args": vars(args),
                            "unet_config": unet_config,
                        }
                        os.makedirs(ckpt_path, exist_ok=True)
                        with open(os.path.join(ckpt_path, "meta.json"), "w") as f:
                            json.dump(meta, f, indent=2, sort_keys=True)

        if global_step >= args.max_train_steps:
            break

    # accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f"Training finished at step {global_step}.")

    # save final accelerate state
    ckpt_root = os.path.join(args.output_dir, "ckpt")
    final_dir = os.path.join(ckpt_root, f"step{global_step}_final")
    accelerator.print(f"[CKPT] Saving final accelerate state to {final_dir}")
    accelerator.save_state(final_dir)

    if accelerator.is_main_process:
        meta = {
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "unet_config": unet_config,
        }
        os.makedirs(final_dir, exist_ok=True)
        with open(os.path.join(final_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
