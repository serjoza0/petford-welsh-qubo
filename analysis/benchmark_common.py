import os
import sys
import glob
import csv
from time import perf_counter
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scripts import CSRGraph, petford_welsh_jit, read_edge_list, create_networkx_graph

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

DEFAULT_A = 1
DEFAULT_B = 1.5
DEFAULT_BASE = 18
DEFAULT_SEED = 0
DEFAULT_MAX_ITERS = 10000
DEFAULT_NUM_ATTEMPTS = 50

ATTEMPT_BUCKETS = [(130, 500), (250, 500), (600, 200), (float("inf"), 100)]
QUICK_ATTEMPT_BUCKETS = [(130, 5), (250, 5), (600, 3), (float("inf"), 2)]


def instances_dir(base_dir=None):
    return os.path.join(base_dir or os.getcwd(), "instances", "stable_set")

def instance_path(name, base_dir=None):
    return os.path.join(instances_dir(base_dir), f"{name}_stable_set_edge_list.txt")

def all_instance_names(base_dir=None):
    paths = sorted(glob.glob(os.path.join(instances_dir(base_dir), "*.txt")))
    return [os.path.basename(p).replace("_stable_set_edge_list.txt", "") for p in paths]

def load_instance(name, base_dir=None):
    return CSRGraph.from_edge_list_file(instance_path(name, base_dir), name=name)

def load_instance_nx(name, base_dir=None):
    n, edges = read_edge_list(instance_path(name, base_dir))
    return create_networkx_graph(n, edges)

def iter_graphs(names=None, base_dir=None, max_n=None, verbose=True):
    names = names or all_instance_names(base_dir)
    total = len(names)
    for idx, name in enumerate(names):
        path = instance_path(name, base_dir)
        if not os.path.exists(path):
            if verbose:
                print(f"[{idx+1}/{total}] {name}: instance file not found, skipping")
            continue
        graph = CSRGraph.from_edge_list_file(path, name=name)
        if max_n is not None and graph.n > max_n:
            if verbose:
                print(f"[{idx+1}/{total}] {name}: n={graph.n} > max-n={max_n}, skipping")
            continue
        yield idx, total, name, graph

def attempts_for(n, buckets=ATTEMPT_BUCKETS):
    for max_n, num_attempts in buckets:
        if n <= max_n:
            return num_attempts
    raise RuntimeError("Check value of n")

def jit_warmup(graph, A=DEFAULT_A, B=DEFAULT_B, b=DEFAULT_BASE, seed=1):
    petford_welsh_jit(graph, A=A, B=B, b=b, max_iters=10, rng=np.random.default_rng(seed))

def make_annealing_schedule(max_iters, b_lo, b_hi, kind="geometric"):
    if kind == "geometric":
        return np.geomspace(b_lo, b_hi, num=max_iters)
    if kind == "linear":
        return np.linspace(b_lo, b_hi, num=max_iters)
    raise ValueError(f"unknown schedule kind: {kind!r}")

def run_sa_attempts(nx_graph, num_attempts, beta, sampler):
    from qpu_comparison import calculate_best_solution, eliminate_and_recalculate
    out = []
    for _ in range(num_attempts):
        t0 = perf_counter()
        solutions = calculate_best_solution(
            nx_graph, sampler, beta=beta, num_of_runs=1,
            no_output_file=True, console_output=False,
        )
        sample_set = solutions[0]["sample_set"]
        rec = eliminate_and_recalculate(nx_graph, sample_set, beta, sampler=sampler, num_of_runs=1, console_output=False)
        elapsed = perf_counter() - t0
        best = len(rec["best_recalculated_solution"]["recalculated_solution_nodes"])
        out.append((best, elapsed))
    return out

def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def results_path(filename, base_dir=None):
    return os.path.join(base_dir or os.getcwd(), "results", filename)

def add_instance_arg(parser, default="C125.9"):
    parser.add_argument("--instance", default=default)
    return parser

def add_instances_arg(parser):
    parser.add_argument("--instances", nargs="+", default=None,
                         help="specific instance names; default is every instance in instances/stable_set")
    return parser

def add_solver_arg(parser, default="jit"):
    parser.add_argument("--solver", choices=["python", "jit"], default=default)
    return parser

def add_pw_args(parser, default_max_iters=DEFAULT_MAX_ITERS, default_num_attempts=DEFAULT_NUM_ATTEMPTS, default_seed=DEFAULT_SEED):
    parser.add_argument("--max-iters", type=int, default=default_max_iters)
    parser.add_argument("--num-attempts", type=int, default=default_num_attempts)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--A", type=float, default=DEFAULT_A)
    parser.add_argument("--B", type=float, default=DEFAULT_B)
    parser.add_argument("--base", type=float, default=DEFAULT_BASE)
    return parser

def add_skip_sa_arg(parser):
    parser.add_argument("--skip-sa", action="store_true")
    return parser