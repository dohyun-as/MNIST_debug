"""Verify that Accelerator built with new config drops RNG broadcast."""
import os, torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils.dataclasses import DistributedType


class Tiny(Dataset):
    def __init__(self): self.x = torch.randn(512, 4)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


def main():
    os.environ["ACCELERATE_USE_CPU"] = "1"
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    acc = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        rng_types=[],
    )
    acc.state.distributed_type = DistributedType.MULTI_CPU
    acc.state.num_processes = dist.get_world_size()
    acc.state.process_index = dist.get_rank()
    acc.state.local_process_index = int(os.environ.get("LOCAL_RANK", 0))
    loader = DataLoader(Tiny(), batch_size=16, shuffle=True, drop_last=True)
    loader = acc.prepare(loader)
    rank = acc.process_index
    print(f"[rank {rank}] rng_types={loader.rng_types} "
          f"sync_gen={loader.synchronized_generator} "
          f"sampler_type={type(loader.batch_sampler.batch_sampler.sampler).__name__ if hasattr(loader, 'batch_sampler') and loader.batch_sampler else 'N/A'}")
    # Run a few epochs to ensure no broadcast hangs
    for ep in range(5):
        first = None
        for batch in loader:  # exhaust to advance epoch counter
            if first is None:
                first = batch
        r0 = first[0, 0].item()
        gathered = [torch.zeros(1) for _ in range(acc.num_processes)]
        dist.all_gather(gathered, torch.tensor([r0]))
        if rank == 0:
            vals = [g.item() for g in gathered]
            print(f"  ep{ep}: first_across_ranks={vals} all_differ={len(set(vals))==acc.num_processes}")


if __name__ == "__main__":
    main()
