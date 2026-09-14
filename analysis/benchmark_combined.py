import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scripts import *
from benchmark_common import KNOWN_ALPHA, DEFAULT_A, DEFAULT_B, DEFAULT_BASE, DEFAULT_MAX_ITERS, DEFAULT_NUM_ATTEMPTS, DEFAULT_SEED

def geometric(lo, hi, max_iters):
    return np.geomspace(lo, hi, num=max_iters)


def linear(lo, hi, max_iters):
    return np.linspace(lo, hi, num=max_iters)


B_SCHEDULES = {
    "constant (8)":              lambda mi: np.full(mi, float(DEFAULT_BASE)),
    "geometric hot->cold (2-100)": lambda mi: geometric(2, 100, mi),
    "geometric cold->hot (100-2)": lambda mi: geometric(100, 2, mi),
    "linear hot->cold (2-100)":    lambda mi: linear(2, 100, mi),
}

RATIO_SCHEDULES = {
    "constant (2.0)":            lambda mi: np.full(mi, float(DEFAULT_B)),
    "geometric hot->cold (1-5)": lambda mi: geometric(1, 5, mi),
    "geometric cold->hot (5-1)": lambda mi: geometric(5, 1, mi),
    "linear hot->cold (1-5)":    lambda mi: linear(1, 5, mi),
}


def run_attempts(graph, b_arr, ratio_arr, max_iters, num_attempts, seed):
    vals = []
    times = []
    for i in range(num_attempts):
        rng = np.random.default_rng(seed + i)
        _, best_value, _, elapsed = petford_welsh_jit(
            graph, A=DEFAULT_A, B=ratio_arr, b=b_arr, max_iters=max_iters, rng=rng,
        )
        vals.append(best_value)
        times.append(elapsed)
    return vals, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=list(KNOWN_ALPHA))
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--num-attempts", type=int, default=DEFAULT_NUM_ATTEMPTS)
    parser.add_argument("--b-schedules", nargs="+", choices=list(B_SCHEDULES), default=list(B_SCHEDULES))
    parser.add_argument("--ratio-schedules", nargs="+", choices=list(RATIO_SCHEDULES), default=list(RATIO_SCHEDULES))
    parser.add_argument("--no-annotate", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    unknown = [name for name in args.instances if name not in KNOWN_ALPHA]
    if unknown:
        raise ValueError(f"no known alpha for: {unknown}")

    n_cells = len(args.b_schedules) * len(args.ratio_schedules)
    total_runs = n_cells * len(args.instances) * args.num_attempts
    print(f"instances={len(args.instances)}  b_schedules={len(args.b_schedules)}  "
          f"ratio_schedules={len(args.ratio_schedules)}  num_attempts={args.num_attempts}  max_iters={args.max_iters}")
    print(f"-> {total_runs} total PW runs")
    if not args.yes:
        resp = input("proceed? [y/N] ").strip().lower()
        if resp != "y":
            print("aborted.")
            return

    base_dir = os.getcwd()

    warm_path = os.path.join(base_dir, "instances", "stable_set", f"{args.instances[0]}_stable_set_edge_list.txt")
    warm_graph = CSRGraph.from_edge_list_file(warm_path, name=args.instances[0])
    petford_welsh_jit(warm_graph, A=DEFAULT_A, B=DEFAULT_B, b=DEFAULT_BASE, max_iters=10, rng=np.random.default_rng(1))

    nb = len(args.b_schedules)
    nr = len(args.ratio_schedules)
    rel_gap_sum = np.zeros((nb, nr))
    n_ok = 0
    t_start = time.perf_counter()

    for idx, name in enumerate(args.instances):
        target = KNOWN_ALPHA[name]
        path = os.path.join(base_dir, "instances", "stable_set", f"{name}_stable_set_edge_list.txt")
        try:
            graph = CSRGraph.from_edge_list_file(path, name=name)
        except FileNotFoundError:
            print(f"[{idx+1}/{len(args.instances)}] {name}: instance file not found, skipping")
            continue

        print(f"[{idx+1}/{len(args.instances)}] {name} (n={graph.n}, m={graph.m}, alpha={target})", end="", flush=True)
        t0 = time.perf_counter()
        for i, b_name in enumerate(args.b_schedules):
            b_arr = B_SCHEDULES[b_name](args.max_iters)
            for j, r_name in enumerate(args.ratio_schedules):
                ratio_arr = RATIO_SCHEDULES[r_name](args.max_iters)
                vals, _ = run_attempts(graph, b_arr, ratio_arr, args.max_iters, args.num_attempts, DEFAULT_SEED)
                avg_best = float(np.mean(vals))
                rel_gap_sum[i, j] += (target - avg_best) / target
        n_ok += 1
        print(f" -- {time.perf_counter() - t0:.1f}s")

    total_elapsed = time.perf_counter() - t_start
    print(f"\ndone: {n_ok}/{len(args.instances)} instances in {total_elapsed:.1f}s")
    if n_ok == 0:
        print("no instances succeeded, nothing to plot.")
        return

    mean_rel_gap = rel_gap_sum / n_ok

    print(f"\nmean rel_gap over {n_ok} instances:")
    print(f"{'b_schedule':30s} {'ratio_schedule':28s} {'mean_rel_gap':>13s}")
    for i, b_name in enumerate(args.b_schedules):
        for j, r_name in enumerate(args.ratio_schedules):
            print(f"{b_name:30s} {r_name:28s} {mean_rel_gap[i, j]:13.4f}")

    fig, ax = plt.subplots(figsize=(1.6 * nr + 3, 1.2 * nb + 3))
    im = ax.imshow(mean_rel_gap, origin="lower", cmap="viridis_r", aspect="equal")

    ax.set_xticks(range(nr))
    ax.set_xticklabels(args.ratio_schedules, rotation=30, ha="right")
    ax.set_yticks(range(nb))
    ax.set_yticklabels(args.b_schedules)
    ax.set_xlabel("ratio (B/A) schedule")
    ax.set_ylabel("b (temperature) schedule")
    ax.set_title(f"cross product of b x ratio schedules -- mean rel. gap over {n_ok} instances")

    if "constant (8)" in args.b_schedules and "constant (2.0)" in args.ratio_schedules:
        bi = args.b_schedules.index("constant (8)")
        rj = args.ratio_schedules.index("constant (2.0)")
        ax.add_patch(plt.Rectangle((rj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))

    if not args.no_annotate:
        vmin, vmax = float(np.nanmin(mean_rel_gap)), float(np.nanmax(mean_rel_gap))
        mid = (vmin + vmax) / 2
        for i in range(nb):
            for j in range(nr):
                color = "white" if mean_rel_gap[i, j] < mid else "black"
                ax.text(j, i, f"{mean_rel_gap[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean rel. gap ((alpha - best)/alpha)")

    out_path = os.path.join(base_dir, "results", "combined_schedule_cross_product.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()