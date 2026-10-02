"""Benchmark of the unified TOP label computation on synthetic lidar sequences.

Examples:
    python -m top_label.benchmark --device cpu --rings 16 --az 360
    python -m top_label.benchmark --device cuda --rings 32 --az 1080 --repeat 5   # nuScenes-like, ~34k points
    python -m top_label.benchmark --device cuda --prune none --rings 16 --az 360  # dense baseline
"""
import argparse
import time

import torch

from top_label import TopLabelConfig, compute_top_labels
from top_label.synthetic import make_sequence


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rings", type=int, default=32)
    ap.add_argument("--az", type=int, default=1080)
    ap.add_argument("--n-past", type=int, default=6)
    ap.add_argument("--n-future", type=int, default=6)
    ap.add_argument("--step", type=float, default=3.0, help="ego displacement per frame [m]")
    ap.add_argument("--noise", type=float, default=0.02, help="range noise std [m]")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--prune", default="phi", choices=["phi", "none"])
    ap.add_argument("--pair-chunk", type=int, default=2_000_000)
    ap.add_argument("--sampling", default="paper", choices=["paper", "uniform"])
    ap.add_argument("--theta", type=float, default=0.003)
    ap.add_argument("--w-min", type=float, default=0.1)
    ap.add_argument("--num-samples", type=int, default=34000, help="dataset size for the extrapolation")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    cfg = TopLabelConfig(theta_dvg=args.theta, w_min=args.w_min, prune=args.prune,
                         pair_chunk=args.pair_chunk, sampling=args.sampling)
    co, cp, no, np_ = make_sequence(n_rings=args.rings, n_az=args.az, n_past=args.n_past,
                                    n_future=args.n_future, step=args.step, range_noise=args.noise)
    co, cp = co.to(device), cp.to(device)
    no = [o.to(device) for o in no]
    np_ = [p.to(device) for p in np_]
    n_pts = cp.shape[0]
    n_nbr_pts = sum(p.shape[0] for p in np_)
    print(f"device={device} torch={torch.__version__} threads={torch.get_num_threads()}")
    print(f"current points N={n_pts}, neighbor frames T={len(np_)}, neighbor points total={n_nbr_pts}")
    print(f"config: {cfg}")

    # warm-up
    compute_top_labels(co, cp, no, np_, cfg)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    stats = {}
    out = None
    for _ in range(args.repeat):
        stats = {}
        _sync(device)
        t0 = time.perf_counter()
        out = compute_top_labels(co, cp, no, np_, cfg, stats=stats)
        _sync(device)
        times.append(time.perf_counter() - t0)

    t_med = sorted(times)[len(times) // 2]
    dense_pairs = n_pts * n_nbr_pts
    n_out = out["cur_idx"].numel()
    counts = torch.bincount(out["state"].long(), minlength=3).tolist()
    print(f"time per current scan: median {t_med:.3f} s (runs: {', '.join(f'{t:.3f}' for t in times)})")
    print(f"candidate pairs: {stats.get('candidates', 0):,} ({100.0 * stats.get('candidates', 0) / dense_pairs:.3f} % of N x M = {dense_pairs:,})")
    print(f"output samples: {n_out:,} (unknown {counts[0]:,}, free {counts[1]:,}, occupied {counts[2]:,}), "
          f"current beams with >= 1 sample: {torch.unique(out['cur_idx']).numel():,}")
    if device.type == "cuda":
        print(f"peak GPU memory allocated: {torch.cuda.max_memory_allocated(device) / 2 ** 20:.1f} MiB")
    total_h = t_med * args.num_samples / 3600.0
    print(f"extrapolated single-device time for {args.num_samples} scans: {total_h:.2f} h "
          f"(compute only; data loading not included)")


if __name__ == "__main__":
    main()
