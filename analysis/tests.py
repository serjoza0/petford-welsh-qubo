import numpy as np
import networkx as nx

from scripts import *

def check_state_consistency(seed=0, n=30, p=0.15, n_steps=800):
    rng = np.random.default_rng(seed)
    G = nx.gnp_random_graph(n, p, seed=seed)
    graph = CSRGraph.from_networkx(G, name="rand")
    problem = MaxStableSetProblem(graph)
    state = problem.initial_state(rng)

    failures = []
    for step in range(n_steps):
        u = int(rng.integers(0, n))
        new_val = int(rng.integers(0,2))
        problem.apply(state, u, new_val)

        true_neighbor_sum = np.array(
            [int(state[list(neighbors(graph, v))].sum()) for v in range(n)]
        )
        if not np.array_equal(problem.neighbor_sum, true_neighbor_sum):
            failures.append((step, "neighbor_sum_mismatch"))
            break

        true_conflicted = {
            v for v in range(n) if state[v] == 1 and true_neighbor_sum[v] > 0
        }
        if problem.conflicted != true_conflicted:
            failures.append((step, "conflicted_set_mismatch"))
            break

        true_energy = -problem.A * state.sum() + problem.B * 0.5 * np.dot(state, true_neighbor_sum)
        if abs(problem.energy(state) - true_energy) > 1e-9:
            failures.append((step, "energy mismatch"))
            break

        true_feasible = len(true_conflicted) == 0
        if problem.is_feasible(state) != true_feasible:
            failures.append("is_feasible mismatch")
            break

    if failures:
        print(f"  FAILED at step {failures[0][0]}: {failures[0][1]}")
        return False
    print(f"  OK: {n_steps} random apply() steps, all invariants held.")
    return True

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

def check_optimality(seed=0, sizes=(8,10,12,14), p=0.35, num_attempts=100, max_iters=1500):
    rng = np.random.default_rng(seed)
    all_ok = True
    for n in sizes:
        G = nx.gnp_random_graph(n, p, seed=seed + n)
        graph = CSRGraph.from_networkx(G, name=f"tiny_{n}")
        true_alpha, _ = brute_force_alpha(graph)

        best_found = 0
        for _ in range(num_attempts):
            problem = MaxStableSetProblem(graph, A=1.0, B=2.0)
            petford_welsh(problem, b=4.0, max_iters=max_iters, rng=rng, record_every=max_iters)
            if problem.best_state is not None:
                best_found = max(best_found, int(problem.best_state.sum()))

        status = "OK" if best_found <= true_alpha else "BUG (exceeded true optimum!)"
        hit = "reached optimum" if best_found == true_alpha else f"gap={true_alpha - best_found}"
        print(f"  n={n:3d}: true alpha={true_alpha}, heuristic best={best_found}  [{status}, {hit}]")
        # nx.draw(G); plt.show()
        if best_found > true_alpha:
            all_ok = False
        if best_found < true_alpha:
            all_ok = False
    if all_ok:
        print("  OK: heuristic matched brute-force optimum on all tiny graphs, never exceeded it.")
    else:
        print("  FAILED: see per-graph detail above.")
    return all_ok


if __name__ == "__main__":
    check_state_consistency()
    check_optimality()