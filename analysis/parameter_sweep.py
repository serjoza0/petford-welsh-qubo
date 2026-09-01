import numpy as np
import networkx as nx
import os
import csv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from scripts import *

INSTANCES = {
    # "hamming6_2": "instances/stable_set/hamming6_2_stable_set_edge_list.txt",
    # "hamming6_4": "instances/stable_set/hamming6_4_stable_set_edge_list.txt",
    # "paley61":    "instances/stable_set/paley61_stable_set_edge_list.txt",
    # "dsjc125.9":  "instances/stable_set/dsjc125.9_stable_set_edge_list.txt",
    # "keller4":    "instances/stable_set/keller4_stable_set_edge_list.txt",
    # "MANN_9":     "instances/stable_set/MANN_a9_stable_set_edge_list.txt",
    # "C125.9":     "instances/stable_set/C125.9_stable_set_edge_list.txt",
    # "evil_myc5x24": "instances/stable_set/evil-N120-p98-myc5x24_stable_set_edge_list.txt",
    "brock800_4": "instances/stable_set/brock800_4_stable_set_edge_list.txt",
    "p_hat1500_3": "instances/stable_set/p_hat1500_3_stable_set_edge_list.txt",
    "sanr200-0-7": "instances/stable_set/sanr200-0-7_stable_set_edge_list.txt",
    "c-fat500-5": "instances/stable_set/c-fat500-5_stable_set_edge_list.txt",
}

TRUE_VALUES = {
    "hamming6_2": 32,
    "hamming6_4": 4,
    # "paley61": 5,
    "dsjc125.9": 34,
    # "keller4": 11,
    # "MANN_9": 16,
    "C125.9": 34,
    "evil_myc5x24": 48,
    "brock800_4": 26,
    "p_hat1500_3": 94,
    "sanr200-0-7": 30,
    "c-fat500-5": 64,
}

SOLVER = "jit"
A = 1.0
B_RATIOS = [1.5, 2, 3, 5, 7, 10, 15, 20]
BASES = [3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20]
NUM_ATTEMPTS = 100
MAX_ITERS = 100000
SEED = 0

def main():
    base_dir = os.getcwd()
    rng = np.random.default_rng(SEED)
    rows = []

    for inst_name, rel_path in INSTANCES.items():
        path = os.path.join(base_dir, rel_path)
        graph = CSRGraph.from_edge_list_file(path, name=inst_name)
        print(f"\n--- {inst_name} (n={graph.n}, m={graph.m}) ---")

        for ratio in B_RATIOS:
            B = A * ratio
            for base in BASES:
                result = run_multi_start(
                    graph, solver=SOLVER, A=A, B=B, b=base,
                    max_iters=MAX_ITERS, num_attempts=NUM_ATTEMPTS, seed=SEED,
                )
                rows.append({
                    "instance": inst_name, "n": graph.n, "m": graph.m,
                    "A": A, "B": B, "ratio": ratio, "base": base,
                    "best": result.best, "mean": result.mean, "std": result.std,
                    "sizes": result.sizes
                })   
                print(f"  ratio={ratio:>4} base={base:>3}: best={result.best:3d} mean={result.mean:6.2f} std={result.std:5.2f}")

    csv_path = os.path.join(base_dir, "results", "param_sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
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
    print(f"\nRecommended default: A=1.0, B={A*best_ratio}, base b={best_base}")


if __name__ == "__main__":
    main()
