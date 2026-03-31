import os, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local_rank)
x = torch.ones(1024, device="cuda") * (rank + 1)

for i in range(1000):
    dist.all_reduce(x)
torch.cuda.synchronize()

if rank == 0:
    print("OK, all_reduce done:", x[0].item())
dist.destroy_process_group()
