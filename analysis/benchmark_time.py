import os
import sys
import time
import numpy as np
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scripts import CSRGraph, MaxStableSetProblem, petford_welsh, read_edge_list, create_networkx_graph

from dwave.samplers import SimulatedAnnealingSampler
from qpu_comparison import calculate_best_solution, eliminate_and_recalculate

KNOWN_ALPHA = {
    "C125.9": 34, "MANN_a9": 16, "brock200-4": 17,
    "brock800_1": 23, "brock800_2": 24, "brock800_3": 25, "brock800_4": 26,
    "c-fat200-1": 12, "c-fat200-2": 24, "c-fat200-5": 58,
    "c-fat500-1": 14, "c-fat500-2": 26, "c-fat500-5": 64,
    "dsjc125.5": 10, "dsjc125.9": 34,
    "evil-N120-p98-chv12x10": 20, "evil-N120-p98-myc5x24": 48,
    "evil-N121-p98-myc11x11": 22, "evil-N125-p98-s3m25x5": 20,
    "hamming6_2": 32, "hamming6_4": 4,
    "johnson16_2_4": 8, "johnson8_2_4": 4, "johnson8_4_4": 14,
    "keller4": 11, "p-hat500-1": 9,
    "p_hat1500_1": 12, "p_hat1500_2": 65, "p_hat1500_3": 94,
    "paley101": 5, "paley61": 5, "paley73": 5, "paley89": 5, "paley97": 6,
    "san200-0-7-1": 30, "san200-0-7-2": 18, "sanr200-0-7": 18,
}



NUM_RESTARTS = 20        # your PW attempts, and SA's num_reads
PW_MAX_ITERS = 5000
SA_BETA = 0.5             # the paper's own recommended penalty value
SA_POST_RUNS = 100        # matches the paper's S_post budget
SEED = 7

base_dir = os.getcwd()
inst_dir = os.path.join(base_dir, "instances", "stable_set")
paths = sorted(glob.glob(os.path.join(inst_dir, "*.txt")))

INSTANCES = {os.path.splitext(os.path.basename(path))[0].replace("_stable_set_edge_list", ""): path for path in paths}


def run_petford_welsh(graph: CSRGraph, num_restarts: int, max_iters: int, rng: np.random.Generator):
    t0 = time.perf_counter()
    best = 0
    for _ in range(num_restarts):
        problem = MaxStableSetProblem(graph, A=1.0, B=2.0)
        target = KNOWN_ALPHA.get(graph.name)
        petford_welsh(problem, b=4.0, max_iters=max_iters, rng=rng, record_every=max_iters, target=target) #type: ignore
        if problem.best_state is not None:
            best = max(best, int(problem.best_state.sum()))
        if target is not None and problem.best_value >= target:
            break

    elapsed = time.perf_counter() - t0
    return best, elapsed


def run_reference_sa(nx_graph, num_runs: int, beta: float, post_runs: int):
    sampler = SimulatedAnnealingSampler()

    t0 = time.perf_counter()
    solutions = calculate_best_solution(
        nx_graph, sampler, beta=beta, num_of_runs=num_runs,
        no_output_file=True, console_output=False,
    )
    raw_elapsed = time.perf_counter() - t0

    sample_set = solutions[0]["sample_set"]
    t0 = time.perf_counter()
    rec = eliminate_and_recalculate(
        nx_graph, sample_set, beta, sampler=sampler,
        num_of_runs=post_runs, console_output=False,
    )
    post_elapsed = time.perf_counter() - t0

    best = len(rec["best_recalculated_solution"]["recalculated_solution_nodes"])
    return best, raw_elapsed, post_elapsed


def main():
    base_dir = os.getcwd()
    rng = np.random.default_rng(SEED)

    header = (f"{'instance':24s} {'known':>6s} | {'PW best':>8s} {'PW time':>8s} | "
              f"{'SA best':>8s} {'SA raw':>8s} {'SA post':>8s} {'SA total':>9s}")
    print(header)
    print("-" * len(header))

    for name in INSTANCES:
        path = os.path.join(base_dir, "instances", "stable_set", f"{name}_stable_set_edge_list.txt")
        n, edges = read_edge_list(path)
        nx_graph = create_networkx_graph(n, edges)
        csr_graph = CSRGraph.from_edge_list_file(path, name=name)

        pw_best, pw_time = run_petford_welsh(csr_graph, NUM_RESTARTS, PW_MAX_ITERS, rng)
        sa_best, sa_raw_time, sa_post_time = run_reference_sa(nx_graph, NUM_RESTARTS, SA_BETA, SA_POST_RUNS)
        sa_total_time = sa_raw_time + sa_post_time

        known = KNOWN_ALPHA.get(name, "?")
        print(f"{name:24s} {str(known):>6s} | {pw_best:8d} {pw_time:7.4f}s | "
              f"{sa_best:8d} {sa_raw_time:7.4f}s {sa_post_time:7.4f}s {sa_total_time:8.4f}s")


if __name__ == "__main__":
    main()