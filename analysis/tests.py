import numpy as np
import networkx as nx
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from scripts import *

def brute_force_alpha(graph: CSRGraph):
    n = graph.n
    edges = []
    for u in range(n):
        for v in neighbors(graph, u):
            if u < v:
                edges.append((u, v))
    best = 0
    best_mask = 0
    for mask in range(1 << n):
        ok = True
        for (u, v) in edges:
            if (mask >> u) & 1 and (mask >> v) & 1:
                ok = False
                break
        if ok:
            c = bin(mask).count("1")
            if c > best:
                best = c
                best_mask = mask
    return best, best_mask

def check_heuristic_optimality(seed=0, sizes=(8, 10, 12, 14), p=0.35, num_attempts=100, max_iters=1500):
    all_ok = True
    for n in sizes:
        G = nx.gnp_random_graph(n, p, seed=seed + n)
        graph = CSRGraph.from_networkx(G, name=f"tiny_{n}")
        true_alpha, _ = brute_force_alpha(graph)

        result = run_multi_start(
            graph, A=1.0, B=2.0, b=4.0,
            max_iters=max_iters, num_attempts=num_attempts, seed=seed,
            known_alpha=true_alpha,
        )
        best_found = result.best

        status = "OK" if best_found <= true_alpha else "BUG (exceeded true optimum!)"
        hit = "reached optimum" if best_found == true_alpha else f"gap={true_alpha - best_found}"
        print(f"  n={n:3d}: true alpha={true_alpha}, best={best_found}  [{status}, {hit}]")
        if best_found != true_alpha:
            all_ok = False
    if all_ok:
        print(f"  OK: heuristic matched brute-force optimum on all tiny graphs, never exceeded it.")
    else:
        print("  FAILED: see per-graph detail above.")
    return all_ok

def check_jit_feasibility(seeds=(0, 1, 2), max_iters=20000):
    print("=== JIT output is actually a stable set ===")
    instances = ["dsjc125.9", "keller4", "brock200-4"]
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    all_ok = True
    for name in instances:
        path = os.path.join(base_dir, "instances", "stable_set", f"{name}_stable_set_edge_list.txt")
        graph = CSRGraph.from_edge_list_file(path, name=name)
        for seed in seeds:
            best_state, best_value, best_step, _ = petford_welsh_jit(
                graph, A=1.0, B=2.0, b=4.0, max_iters=max_iters, rng=np.random.default_rng(seed)
            )
            members = np.flatnonzero(best_state)
            bad_edges = 0
            for v in members:
                for u in neighbors(graph, v):
                    if best_state[u] == 1:
                        bad_edges += 1
            if bad_edges != 0 or len(members) != best_value:
                print(f"  FAILED {name} seed={seed}: bad_edges={bad_edges}, |members|={len(members)}, best_value={best_value}")
                all_ok = False
    if all_ok:
        print(f"  OK: every returned best_state was a genuine independent set on {len(instances)} instances x {len(seeds)} seeds.")
    return all_ok

if __name__ == "__main__":
    check_heuristic_optimality()
    check_jit_feasibility()