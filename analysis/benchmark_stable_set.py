import os
import re
import argparse
import glob
import time
import csv
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from scripts import *

KNOWN_ALPHA = {
    "C125.9": 34,
    "MANN_a9": 16,
    "brock200-4": 17,
    "brock800_1": 23,
    "brock800_2": 24,
    "brock800_3": 25,
    "brock800_4": 26,
    "c-fat200-1": 12,
    "c-fat200-2": 24,
    "c-fat200-5": 58,
    "c-fat500-1": 14,
    "c-fat500-2": 26,
    "c-fat500-5": 64,
    "dsjc125.5": 10,
    "dsjc125.9": 34,
    "evil-N120-p98-chv12x10": 20,
    "evil-N120-p98-myc5x24": 48,
    "evil-N121-p98-myc11x11": 22,
    "evil-N125-p98-s3m25x5": 20,
    "hamming6_2": 32,
    "hamming6_4": 4,
    "johnson16_2_4": 8,
    "johnson8_2_4": 4,
    "johnson8_4_4": 14,
    "keller4": 11,
    "p-hat500-1": 9,
    "p_hat1500_1": 12,
    "p_hat1500_2": 65,
    "p_hat1500_3": 94,
    "paley101": 5,
    "paley61": 5,
    "paley73": 5,
    "paley89": 5,
    "paley97": 6,
    "san200-0-7-1": 30,
    "san200-0-7-2": 18,
    "sanr200-0-7": 18,
}

BUCKETS = [
    # (max_n_inclusive, label, max_iters, num_attempts)
    (130, "small", 3000, 50),
    (250, "medium", 5000, 30),
    (600, "large", 10000, 15),
    (float("inf"), "xlarge", 20000, 8),
]

QUICK_BUCKETS = [
    (130, "small", 500, 5),
    (250, "medium", 500, 5),
    (600, "large", 500, 3),
    (float("inf"), "xlarge", 500, 2),
]

def bucket_for(n, buckets):
    for max_n, label, max_iters, num_attempts in buckets:
        if n <= max_n:
            return label, max_iters, num_attempts
    raise RuntimeError("Check value of n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", type=float, default=1.0)
    parser.add_argument("--B", type=float, default=2.0)
    parser.add_argument("--base", type=float, default=4.0)
    parser.add_argument("--filter", type=str, default=None, help="regex to filter instance filenames")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    base_dir = os.getcwd()
    inst_dir = os.path.join(base_dir, "instances", "stable_set")
    paths = sorted(glob.glob(os.path.join(inst_dir, "*.txt")))

    rng = np.random.default_rng(args.seed)
    buckets = QUICK_BUCKETS if args.quick else BUCKETS

    if args.filter:
        pat = re.compile(args.filter)
        paths = [p for p in paths if pat.search(os.path.basename(p))]

    rows = []
    for path in paths:
        name = os.path.basename(path).replace("_stable_set_edge_list.txt", "")
        graph = CSRGraph.from_edge_list_file(path, name=name)
        label, max_iters, num_attempts = bucket_for(graph.n, buckets)

        if args.list:
            print(f"{name:35s} n={graph.n:5d} m={graph.m:7d}  bucket={label:6s} "
                  f"max_iters={max_iters:6d} attempts={num_attempts}")
            continue

        print(f"[{label:6s}] {name:35s} n={graph.n:5d} m={graph.m:7d} "
              f"(A={args.A}, B={args.B}, base={args.base}, "
              f"max_iters={max_iters}, attempts={num_attempts}) ... ", end="", flush=True)
        
        t0 = time.perf_counter()
        sizes = []
        for _ in range(num_attempts):
            problem = MaxStableSetProblem(graph, A=args.A, B=args.B)
            petford_welsh(problem, b=args.base, max_iters=max_iters, rng=rng, record_every=max_iters)
            if problem.best_state is not None:
                sizes.append(int(problem.best_state.sum()))
            else:
                sizes.append(0)
        elapsed = time.perf_counter() - t0

        best, mean, std = max(sizes), float(np.mean(sizes)), float(np.std(sizes))
        print(f"best={best:4d} mean={mean:7.2f} std={std:5.2f} time={elapsed:6.1f}s")

        rows.append({
            "instance": name, "n": graph.n, "m": graph.m,
            "A": args.A, "B": args.B, "base": args.base,
            "max_iters": max_iters, "num_attempts": num_attempts,
            "best": best, "mean": mean, "std": std,
            "time_sec": elapsed,
            "known_best": KNOWN_ALPHA.get(name, ""),
            "gap": (KNOWN_ALPHA[name] - best) if name in KNOWN_ALPHA else "",
        })

    if args.list or not rows:
        return

    csv_path = os.path.join(base_dir, "results", f"benchmark_stable_set_results{'_quick' if args.quick else ''}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()