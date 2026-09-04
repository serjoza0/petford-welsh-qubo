import numpy as np
import networkx as nx
import os
import csv

from benchmark_common import *
from scripts import *




SOLVER = "jit"
B_RATIOS = [1.5, 2, 3, 5, 7, 10, 15, 20]
BASES = [3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20]

def main():
    base_dir = os.getcwd()
    names = all_instance_names(base_dir)

    rows = []
    for name in names:
        graph = load_instance(name, base_dir)
        print(f"\n--- {name} (n={graph.n}, m={graph.m}) ---")

        for ratio in B_RATIOS:
            B = DEFAULT_A * ratio
            for base in BASES:
                result = run_multi_start(
                    graph, solver=SOLVER, A=DEFAULT_A, B=B, b=base,
                    max_iters=DEFAULT_MAX_ITERS, num_attempts=DEFAULT_NUM_ATTEMPTS, seed=DEFAULT_SEED,
                )
                rows.append({
                    "instance": name, "n": graph.n, "m": graph.m,
                    "A": DEFAULT_A, "B": B, "ratio": ratio, "base": base,
                    "best": result.best, "mean": result.mean, "std": result.std,
                    "sizes": result.sizes
                })   
                print(f"  ratio={ratio:>4} base={base:>3}: best={result.best:3d} mean={result.mean:6.2f} std={result.std:5.2f}")

    csv_path = os.path.join(base_dir, "results", "param_sweep_results.csv")
    write_csv(rows, csv_path)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    per_instance_best = {}
    for r in rows:
        per_instance_best[r["instance"]] = max(
            per_instance_best.get(r["instance"], 0), r["best"]
        )
    agg = {}
    for r in rows:
        rel = r["best"] / per_instance_best[r["instance"]]
        agg.setdefault((r["ratio"], r["base"]), []).append(rel)

    print("\n=== Robustness ranking (avg relative-best across instances) ===")
    ranked = sorted(agg.items(), key=lambda kv: -np.mean(kv[1]))
    for (ratio, base), rels in ranked:
        print(f"  ratio={ratio:>4} base={base:>3}: avg_rel_best={np.mean(rels):.3f} "
              f"min_rel_best={min(rels):.3f}")

    best_ratio, best_base = ranked[0][0]
    print(f"\nRecommended default: A=1.0, B={DEFAULT_A*best_ratio}, base b={best_base}")


if __name__ == "__main__":
    main()
