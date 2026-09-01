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

from dwave.samplers import SimulatedAnnealingSampler
from qpu_comparison import calculate_best_solution, eliminate_and_recalculate

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

ITERS_PER_N = 500
MIN_ITERS = 200
QUICK_ITERS_PER_N = 200
QUICK_MIN_ITERS = 500

ATTEMPT_BUCKETS = [
    (130, 500), (250, 500), (600, 200), (float("inf"), 100),
]
QUICK_ATTEMPT_BUCKETS = [
    (130, 5), (250, 5), (600, 3), (float("inf"), 2),
]

def attempts_for(n, buckets):
    for max_n, num_attempts in buckets:
        if n <= max_n:
            return num_attempts
    raise RuntimeError("Check value of n")

def run_reference_sa(path, name, num_attempts):
    n, edges = read_edge_list(path)
    nx_graph = create_networkx_graph(n, edges)
    sampler = SimulatedAnnealingSampler()

    t0 = time.perf_counter()
    solutions = calculate_best_solution(
        nx_graph, sampler, beta=0.5, num_of_runs=num_attempts,
        no_output_file=True, console_output=False,
    )
    raw_elapsed = time.perf_counter() - t0

    sample_set = solutions[0]["sample_set"]
    t0 = time.perf_counter()
    rec = eliminate_and_recalculate(
        nx_graph, sample_set, sample_beta=0.5, sampler=sampler,
        num_of_runs=100, console_output=False,
    )
    post_elapsed = time.perf_counter() - t0

    sa_best = len(rec["best_recalculated_solution"]["recalculated_solution_nodes"])
    return sa_best, raw_elapsed + post_elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", type=float, default=1.0)
    parser.add_argument("--B", type=float, default=2.0)
    parser.add_argument("--base", type=float, default=4.0)
    parser.add_argument("--filter", type=str, default=None, help="regex to filter instance filenames")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-sa", action="store_true", help="skip the SA reference runs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iters-per-n", type=int, default=None)
    args = parser.parse_args()

    base_dir = os.getcwd()
    inst_dir = os.path.join(base_dir, "instances", "stable_set")
    paths = sorted(glob.glob(os.path.join(inst_dir, "*.txt")))

    attept_buckets = QUICK_ATTEMPT_BUCKETS if args.quick else ATTEMPT_BUCKETS
    iters_per_n = args.iters_per_n  if args.iters_per_n is not None else (QUICK_ITERS_PER_N if args.quick else ITERS_PER_N)
    min_iters = QUICK_MIN_ITERS if args.quick else MIN_ITERS

    if args.filter:
        pat = re.compile(args.filter)
        paths = [p for p in paths if pat.search(os.path.basename(p))]

    results = []
    rows = []
    for path in paths:
        name = os.path.basename(path).replace("_stable_set_edge_list.txt", "")
        graph = CSRGraph.from_edge_list_file(path, name=name)
        num_attempts = attempts_for(graph.n, attept_buckets)
        max_iters = max(min_iters, graph.n * iters_per_n)

        if args.list:
            print(f"{name:35s} n={graph.n:5d} m={graph.m:7d} "
                  f"max_iters={max_iters:6d} attempts={num_attempts}")
            continue

        print(f"{name:35s} n={graph.n:5d} m={graph.m:7d} "
              f"(A={args.A}, B={args.B}, base={args.base}, "
              f"max_iters={max_iters}, attempts={num_attempts}) ... ", end="\n", flush=True)
        
        result = run_multi_start(path, solver="jit", A=args.A, B=args.B, b=args.base,
                                 max_iters=max_iters, num_attempts=num_attempts, seed=args.seed,
                                 known_alpha=KNOWN_ALPHA.get(name))
        print(f"best={result.best:4d} mean={result.mean:7.2f} std={result.std:5.2f} time={result.total_time:6.1f}s")
        results.append(result)
        gap_str = result.gap if result.gap is not None else "?"
        print(f"best={result.best:4d} mean={result.mean:7.2f} std={result.std:5.2f} "
              f"gap={gap_str} time={result.total_time:6.1f}s")

        row = result.to_row()
        if not args.skip_sa:
            sa_best, sa_total_elapsed = run_reference_sa(path, name, num_attempts)
            print(f"  SA best={sa_best:4d} total={sa_total_elapsed:6.1f}s")
            row["sa_best"] = sa_best
            row["sa_total_time_sec"] = sa_total_elapsed
            row["sa_gap"] = (KNOWN_ALPHA[name] - sa_best) if name in KNOWN_ALPHA else None
        rows.append(row)

    if args.list or not rows:
        return

    suffix = "_quick" if args.quick else ""
    csv_path = os.path.join(base_dir, "results", f"benchmark_stable_set_results{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    save_attempts_csv(results, os.path.join(base_dir, "results", f"benchmark_stable_set_attempts{suffix}.csv"))
    print(f"Wrote per-attempt PW detail to results/benchmark_stable_set_attempts{suffix}.csv")

if __name__ == "__main__":
    main()