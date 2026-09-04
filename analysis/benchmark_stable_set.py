import os
import re
import argparse
import numpy as np

from benchmark_common import *
from scripts import *

from dwave.samplers import SimulatedAnnealingSampler
from qpu_comparison import calculate_best_solution, eliminate_and_recalculate

ITERS_PER_N = 5000
MIN_ITERS = 2000
QUICK_ITERS_PER_N = 200
QUICK_MIN_ITERS = 500

def run_reference_sa(name, num_attempts, base_dir=None):
    nx_graph = load_instance_nx(name, base_dir)
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
    add_pw_args(parser)
    add_skip_sa_arg(parser)
    parser.add_argument("--latex", action="store_true", help="save LaTeX table of results")
    args = parser.parse_args()

    base_dir = os.getcwd()
    names = all_instance_names(base_dir)

    results = []
    rows = []
    for name in names:
        graph = load_instance(name, base_dir)

        print(f"{name:35s} n={graph.n:5d} m={graph.m:7d} "
              f"(A={args.A}, B={args.B}, base={args.base}, "
              f"max_iters={args.max_iters}, attempts={args.num_attempts}) ... ", end="\n", flush=True)
        
        result = run_multi_start(graph, solver="jit", A=args.A, B=args.B, b=args.base,
                                 max_iters=args.max_iters, num_attempts=args.num_attempts, seed=args.seed,
                                 known_alpha=KNOWN_ALPHA.get(name))
        print(f"best={result.best:4d} mean={result.mean:7.2f} std={result.std:5.2f} time={result.total_time:6.1f}s")
        results.append(result)
        gap_str = result.gap if result.gap is not None else "?"
        print(f"best={result.best:4d} mean={result.mean:7.2f} std={result.std:5.2f} "
              f"gap={gap_str} time={result.total_time:6.1f}s")

        row = result.to_row()
        if not args.skip_sa:
            sa_best, sa_total_elapsed = run_reference_sa(name, args.num_attempts, base_dir)
            print(f"  SA best={sa_best:4d} total={sa_total_elapsed:6.1f}s")
            row["sa_best"] = sa_best
            row["sa_total_time_sec"] = sa_total_elapsed
            row["sa_gap"] = (KNOWN_ALPHA[name] - sa_best) if name in KNOWN_ALPHA else None
        rows.append(row)

    csv_path = os.path.join(base_dir, "results", f"benchmark_stable_set_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    save_path = os.path.join(base_dir, "results", f"benchmark_stable_set_attempts.csv")
    save_attempts_csv(results, save_path)
    print(f"Wrote per-attempt PW detail to results/benchmark_stable_set_attempts.csv")

    if args.latex:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df = df.drop(columns=["A", "B", "b", "std"])
        latex_table = df.to_latex(index=False, float_format="%.2f", escape=True)
        with open(os.path.join(base_dir, "report", "benchmark_stable_set_table.tex"), "w") as f:
            f.write(latex_table)

if __name__ == "__main__":
    main()