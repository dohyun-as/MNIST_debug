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

from diffusers import DDPMScheduler, DDIMScheduler
from torchvision.utils import make_grid, save_image

from .dataset import MNISTConditionalDataset
from .model import ConditionalUNet
from .sampling import sample_ddim_with_cfg


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

    # model / unet config
    parser.add_argument("--unet_config", type=str, required=True,
                        help="Path to UNet2DConditionModel config JSON file.")

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


@torch.no_grad()
def run_evaluation(
    accelerator: Accelerator,
    model: ConditionalUNet,
    noise_scheduler: DDPMScheduler,
    unet_config: dict,
    args,
    global_step: int,
):
    """
    eval 시점마다 DDIM 샘플러(sample_ddim_with_cfg)를 사용해서
    class-wise 이미지 생성 (0~9 각 클래스별로 samples_per_class개).
    """
    if not accelerator.is_main_process:
        return

    accelerator.print(f"[Eval] Running evaluation at step {global_step}...")

    unwrapped_model = accelerator.unwrap_model(model)
    device = accelerator.device

    # DDIMScheduler를 train DDPMScheduler config에서 생성
    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    image_size = unet_config.get("sample_size", 32)

    num_classes = 10
    n_per_class = args.eval_num_samples_per_class

    # 라벨: [0,0,...,0, 1,1,...,1, ..., 9,9,...,9]
    labels = torch.arange(num_classes, device=device).repeat_interleave(n_per_class)

    # DDIM 샘플링 (CFG는 일단 사용 안 함: guidance_scale=1.0, uncond_cond=None)
    samples = sample_ddim_with_cfg(
        model=unwrapped_model,
        noise_scheduler=ddim_scheduler,
        cond=labels,
        num_inference_steps=args.eval_num_steps,
        guidance_scale=1.0,
        uncond_cond=None,
        eta=0.0,
        generator=None,
    )  # (B, 1, H, W), B = num_classes * n_per_class

    samples = samples.clamp(-1.0, 1.0)
    samples_01 = (samples + 1.0) * 0.5
    samples_01 = samples_01.clamp(0.0, 1.0)

    # grid: class별로 한 줄씩 → nrow = n_per_class
    grid = make_grid(samples_01, nrow=n_per_class, padding=2)

    eval_dir = os.path.join(args.output_dir, "eval_samples")
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, f"step_{global_step}.png")
    save_image(grid, out_path)

    accelerator.print(f"[Eval] Saved class-wise DDIM samples to {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Accelerator 설정: mixed_precision / grad_accum / tracker 백엔드까지 여기서 지정
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        log_with=args.log_with,
    )

    # 시드: 프로세스마다 다르게
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

    # sample_size를 dataset resize에 사용
    image_size = unet_config.get("sample_size", 32)

    if accelerator.is_main_process:
        print(f"Loaded UNet config from {args.unet_config}")
        print(f"Using image_size={image_size} for MNIST dataset.")

    # -----------------------
    # Dataset & DataLoader
    # -----------------------
    train_dataset = MNISTConditionalDataset(
        root=args.data_dir,
        split="train",
        resize_to=image_size,   # 예: 32x32
        pad_to=None,
    )

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
    model = ConditionalUNet(
        num_classes=10,
        class_embed_dim=unet_config.get("cross_attention_dim", 128),
        image_size=image_size,
        image_conditioning=False,
        encoder=None,
        cond_dim=unet_config.get("cross_attention_dim", 128),
        cond_in_channels=1,
        unet_config=unet_config,
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

    # -----------------------
    # Training Loop (step 기반)
    # -----------------------
    for epoch in range(1, num_epochs + 1):
        model.train()

        for step, (images, labels) in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break

            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.shape[0]

            timesteps = torch.randint(
                low=0,
                high=noise_scheduler.config.num_train_timesteps,
                size=(batch_size,),
                device=device,
            ).long()

            noise = torch.randn_like(images)

            noisy_images = noise_scheduler.add_noise(
                original_samples=images,
                noise=noise,
                timesteps=timesteps,
            )

            # class conditioning
            pred_noise = model(noisy_images, timesteps, y=labels)

            loss = F.mse_loss(pred_noise, noise)

            loss = loss / args.grad_accum_steps
            accelerator.backward(loss)

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # 🔹 진행 상황 / loss 계산
                # loss는 grad_accum 나누기 전 값 기준으로 보정
                avg_loss = accelerator.gather(loss.detach()).mean().item() * args.grad_accum_steps

                # 🔹 tqdm progress bar 업데이트
                progress_bar.update(1)
                logs = {
                    "step_loss": f"{avg_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
                progress_bar.set_postfix(**logs)

                # 🔹 accelerator tracker 로깅 (tensorboard / wandb 등)
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
