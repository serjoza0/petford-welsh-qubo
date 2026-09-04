import os
import sys
import glob
import argparse
import csv
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scripts import *
from scripts.solver import petford_welsh
from scripts.problems import MaxStableSetProblem

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

A = 1
B = 1.5
B_SCHEDULE = 18

ITERS_PER_N = 50
MIN_ITERS = 500


def run_attempt(graph, seed, max_iters):
    rng = np.random.default_rng(seed)
    problem = MaxStableSetProblem(graph, A=A, B=B)
    petford_welsh(problem, b=B_SCHEDULE, max_iters=max_iters, rng=rng, record_every=max_iters)
    best_value = int(problem.best_value)
    repaired_state = problem.convert_to_feasible()
    repaired_value = int(repaired_state.sum())
    return best_value, repaired_value


def run_instance(graph, name, target, max_iters, num_attempts, seed):
    best_vals = []
    repaired_vals = []
    best_wins = 0
    repaired_wins = 0
    ties = 0
    attempt_rows = []

    for i in range(num_attempts):
        s = seed + i
        best_value, repaired_value = run_attempt(graph, s, max_iters)
        best_vals.append(best_value)
        repaired_vals.append(repaired_value)

        if best_value > repaired_value:
            winner = "best"
            best_wins += 1
        elif repaired_value > best_value:
            winner = "repaired"
            repaired_wins += 1
        else:
            winner = "tie"
            ties += 1

        attempt_rows.append({
            "instance": name, "seed": s, "best_value": best_value,
            "repaired_value": repaired_value, "winner": winner,
        })

    summary = {
        "instance": name, "n": graph.n, "m": graph.m,
        "max_iters": max_iters, "num_attempts": num_attempts,
        "known_alpha": target if target is not None else "",
        "avg_best": float(np.mean(best_vals)), "avg_repaired": float(np.mean(repaired_vals)),
        "max_best": max(best_vals), "max_repaired": max(repaired_vals),
        "gap_best": (target - float(np.mean(best_vals))) if target is not None else "",
        "gap_repaired": (target - float(np.mean(repaired_vals))) if target is not None else "",
        "best_wins": best_wins, "repaired_wins": repaired_wins, "ties": ties,
    }
    return summary, attempt_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=None,
                         help="specific instance names; default is every *.txt in instances/stable_set")
    parser.add_argument("--iters-per-n", type=int, default=ITERS_PER_N)
    parser.add_argument("--min-iters", type=int, default=MIN_ITERS)
    parser.add_argument("--num-attempts", type=int, default=5)
    parser.add_argument("--max-n", type=int, default=None, help="skip instances larger than this (python solver is slow)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_dir = os.getcwd()
    inst_dir = os.path.join(base_dir, "instances", "stable_set")

    if args.instances:
        names = args.instances
    else:
        paths = sorted(glob.glob(os.path.join(inst_dir, "*.txt")))
        names = [os.path.basename(p).replace("_stable_set_edge_list.txt", "") for p in paths]

    summaries = []
    attempt_rows = []
    t_start = time.perf_counter()

    for idx, name in enumerate(names):
        path = os.path.join(inst_dir, f"{name}_stable_set_edge_list.txt")
        if not os.path.exists(path):
            print(f"[{idx+1}/{len(names)}] {name}: instance file not found, skipping")
            continue

        graph = CSRGraph.from_edge_list_file(path, name=name)
        if args.max_n is not None and graph.n > args.max_n:
            print(f"[{idx+1}/{len(names)}] {name}: n={graph.n} > max-n={args.max_n}, skipping")
            continue

        max_iters = max(args.min_iters, graph.n * args.iters_per_n)
        target = KNOWN_ALPHA.get(name)

        print(f"[{idx+1}/{len(names)}] {name} (n={graph.n}, m={graph.m}, alpha={target}, "
              f"max_iters={max_iters}, attempts={args.num_attempts}) ...", end="", flush=True)
        t0 = time.perf_counter()
        summary, rows = run_instance(graph, name, target, max_iters, args.num_attempts, args.seed)
        elapsed = time.perf_counter() - t0
        print(f" avg_best={summary['avg_best']:.1f} avg_repaired={summary['avg_repaired']:.1f} "
              f"best_wins={summary['best_wins']} repaired_wins={summary['repaired_wins']} "
              f"({elapsed:.1f}s)")

        summaries.append(summary)
        attempt_rows.extend(rows)

    total_elapsed = time.perf_counter() - t_start
    print(f"\ndone: {len(summaries)}/{len(names)} instances in {total_elapsed:.1f}s")
    if not summaries:
        return

    total_best_wins = sum(s["best_wins"] for s in summaries)
    total_repaired_wins = sum(s["repaired_wins"] for s in summaries)
    total_ties = sum(s["ties"] for s in summaries)
    print(f"overall: best wins {total_best_wins}, repaired wins {total_repaired_wins}, ties {total_ties}")

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "repair_vs_best_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {len(summaries)} rows to {summary_path}")
    # attempts_path = os.path.join(results_dir, "repair_vs_best_attempts.csv")
    # with open(attempts_path, "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=list(attempt_rows[0].keys()))
    #     writer.writeheader()
    #     writer.writerows(attempt_rows)
    # print(f"Wrote {len(attempt_rows)} rows to {attempts_path}")


if __name__ == "__main__":
    main()