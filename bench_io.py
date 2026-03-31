"""Benchmark: NAS vs SSD single-file torch.load() throughput."""
import torch, time, os, argparse

def make_dummy(path):
    """Create a dummy .pt file matching real latent cache format."""
    d = {
        'latent': torch.randn(16, 16, 16, dtype=torch.float16),
        'latent_flip': torch.randn(16, 16, 16, dtype=torch.float16),
        'image': torch.randn(3, 256, 256, dtype=torch.float16),
        'image_flip': torch.randn(3, 256, 256, dtype=torch.float16),
    }
    torch.save(d, path)
    print(f"  Created: {path} ({os.path.getsize(path)/1024:.1f} KB)")

def bench_load(path, n_iters):
    """Load the same file n_iters times, return elapsed seconds."""
    # Warmup (fill OS page cache)
    for _ in range(10):
        torch.load(path, map_location='cpu', weights_only=True)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        torch.load(path, map_location='cpu', weights_only=True)
    elapsed = time.perf_counter() - t0
    return elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000, help='number of loads')
    args = parser.parse_args()

    nas_path = '/workspace/NAS/project/MNIST_debug/runs/imagenet_256_injection/latent_cache/_bench_dummy.pt'
    ssd_path = '/tmp/_bench_dummy.pt'

    print("Creating dummy files...")
    make_dummy(nas_path)
    make_dummy(ssd_path)

    print(f"\nBenchmarking {args.n} torch.load() calls per location...\n")

    # --- SSD ---
    t_ssd = bench_load(ssd_path, args.n)
    print(f"SSD (/tmp):  {t_ssd:.2f}s  |  {args.n/t_ssd:.0f} loads/s  |  {t_ssd/args.n*1000:.3f} ms/load")

    # --- NAS ---
    t_nas = bench_load(nas_path, args.n)
    print(f"NAS:         {t_nas:.2f}s  |  {args.n/t_nas:.0f} loads/s  |  {t_nas/args.n*1000:.3f} ms/load")

    print(f"\nNAS / SSD ratio: {t_nas/t_ssd:.1f}x slower")

    # Cleanup
    os.remove(nas_path)
    os.remove(ssd_path)
    print("Cleaned up dummy files.")

if __name__ == '__main__':
    main()
