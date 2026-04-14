"""Minimal repro: just data loading + accelerate RNG sync.

Run with:
  ACCEL_USE_CPU=1 accelerate launch --cpu --num_processes 4 \
      --main_process_port 29555 _debug_rng_sync.py
"""
import os, sys, time, traceback
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator


class TinyDataset(Dataset):
    def __init__(self, n=4096, dim=16):
        self.x = torch.randn(n, dim)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return {"x": self.x[i]}


def main():
    # Force MULTI_CPU mode (gloo backend, no GPU)
    os.environ["ACCELERATE_USE_CPU"] = "1"
    os.environ.setdefault("ACCELERATE_TORCH_DEVICE", "cpu")
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    from accelerate.state import AcceleratorState
    from accelerate.utils.dataclasses import DistributedType
    from accelerate import DataLoaderConfiguration
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        rng_types=[],
    )
    accelerator.state.distributed_type = DistributedType.MULTI_CPU
    accelerator.state.num_processes = dist.get_world_size()
    accelerator.state.process_index = dist.get_rank()
    accelerator.state.local_process_index = int(os.environ.get("LOCAL_RANK", 0))
    rank = accelerator.process_index
    n_proc = accelerator.num_processes
    print(f"[rank {rank}] start, n_proc={n_proc}, dist_type={accelerator.state.distributed_type}", flush=True)

    ds = TinyDataset(n=4096, dim=16)
    loader = DataLoader(ds, batch_size=128, shuffle=True,
                        num_workers=4, pin_memory=False, drop_last=True)

    loader = accelerator.prepare(loader)
    print(f"[rank {rank}] prepared loader: type={type(loader).__name__} "
          f"rng_types={getattr(loader, 'rng_types', None)} "
          f"sync_gen={getattr(loader, 'synchronized_generator', None)}",
          flush=True)

    # Iterate many epochs to mimic ~5000 update steps
    n_epochs = 200
    t0 = time.time()
    last_state_hash = None
    for ep in range(n_epochs):
        try:
            n_batches = 0
            first_ids = []
            for batch in loader:
                n_batches += 1
                if n_batches <= 2:
                    first_ids.append(batch["x"][0, 0].item())
            # Gather first-item-of-first-batch across ranks to verify that
            # each rank sees DIFFERENT shards (not identical data).
            import torch.distributed as _d
            t = torch.tensor([first_ids[0]], dtype=torch.float32)
            gathered = [torch.zeros_like(t) for _ in range(_d.get_world_size())]
            _d.all_gather(gathered, t)
            if rank == 0 and ep % 20 == 0:
                vals = [g.item() for g in gathered]
                all_same = len(set(vals)) == 1
                print(f"[rank {rank}] ep={ep} batches={n_batches} "
                      f"first_vals_across_ranks={vals} all_same={all_same} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
            continue  # skip the old logging below
            # After each epoch, capture the synchronized generator state to
            # see whether it's stable / non-corrupt.
            gen = getattr(loader, "synchronized_generator", None)
            if gen is not None:
                st = gen.get_state()
                h = (int(st.sum().item()), st.numel(), int(st[0]), int(st[-1]))
                if rank == 0 and ep % 20 == 0:
                    print(f"[rank {rank}] ep={ep} batches={n_batches} "
                          f"state_hash={h} elapsed={time.time()-t0:.1f}s",
                          flush=True)
                # Try to set_state on a fresh generator to validate roundtrip.
                test_gen = torch.Generator()
                test_gen.set_state(st.clone())
            else:
                if rank == 0 and ep % 20 == 0:
                    print(f"[rank {rank}] ep={ep} batches={n_batches} (no sync_gen)",
                          flush=True)
        except Exception as e:
            print(f"[rank {rank}] ep={ep} EXCEPTION: {type(e).__name__}: {e}",
                  flush=True)
            traceback.print_exc()
            sys.exit(1)
    if rank == 0:
        print(f"[rank {rank}] DONE {n_epochs} epochs in {time.time()-t0:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
