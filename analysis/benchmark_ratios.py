import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scripts import *

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

DEFAULT_RATIO_GRID = [0.5, 1, 1.5, 2, 3, 5, 8, 12, 20]
DEFAULT_MAX_ITERS_GRID = [20000, 100000, 500000, 2000000]
A = 1
B_SCHEDULE = 18
SEED = 0

SCHEDULE_BUILDERS = {
    "geometric": lambda r_start, r_end, max_iters: np.geomspace(r_start, r_end, num=max_iters),
    "linear":    lambda r_start, r_end, max_iters: np.linspace(r_start, r_end, num=max_iters),
}


def run_attempts(graph, B_arr, max_iters, num_attempts, seed):
    vals = []
    times = []
    for i in range(num_attempts):
        rng = np.random.default_rng(seed + i)
        _, best_value, _, elapsed = petford_welsh_jit_ratio_schedule(
            graph, A=A, B=B_arr, b=B_SCHEDULE, max_iters=max_iters, rng=rng,
        )
        vals.append(best_value)
        times.append(elapsed)
    return vals, times


def run_sweep(graph, ratio_grid, max_iters, num_attempts, seed, build_schedule):
    ng = len(ratio_grid)
    avg_best = np.zeros((ng, ng))
    avg_time = np.zeros((ng, ng))
    for i, r_start in enumerate(ratio_grid):
        for j, r_end in enumerate(ratio_grid):
            B_arr = build_schedule(r_start, r_end, max_iters)
            vals, times = run_attempts(graph, B_arr, max_iters, num_attempts, seed)
            avg_best[i, j] = float(np.mean(vals))
            avg_time[i, j] = float(np.mean(times))
    return avg_best, avg_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="C125.9")
    parser.add_argument("--ratio-grid", type=float, nargs="+", default=DEFAULT_RATIO_GRID)
    parser.add_argument("--max-iters-grid", type=int, nargs="+", default=DEFAULT_MAX_ITERS_GRID)
    parser.add_argument("--num-attempts", type=int, default=10)
    parser.add_argument("--schedules", choices=list(SCHEDULE_BUILDERS), nargs="+",
                         default=["geometric", "linear"])
    parser.add_argument("--no-annotate", action="store_true")
    args = parser.parse_args()

    base_dir = os.getcwd()
    name = args.instance
    target = KNOWN_ALPHA.get(name)
    path = os.path.join(base_dir, "instances", "stable_set", f"{name}_stable_set_edge_list.txt")
    graph = CSRGraph.from_edge_list_file(path, name=name)

    petford_welsh_jit_ratio_schedule(graph, A=A, B=B_SCHEDULE, b=B_SCHEDULE, max_iters=10, rng=np.random.default_rng(1))

    total_runs = len(args.ratio_grid) ** 2 * len(args.schedules) * len(args.max_iters_grid) * args.num_attempts
    print(f"--- {name} (n={graph.n}, m={graph.m}) known alpha={target} ---")
    print(f"sweeping ratio_start x ratio_end over {len(args.ratio_grid)}x{len(args.ratio_grid)} grid, "
          f"max_iters_grid={args.max_iters_grid}, num_attempts={args.num_attempts}, schedules={args.schedules}")
    print(f"-> {total_runs} total PW runs")

    metric_by_max_iters = {}
    for max_iters in args.max_iters_grid:
        metric_by_max_iters[max_iters] = {}
        for schedule_name in args.schedules:
            t0 = time.perf_counter()
            avg_best, avg_time = run_sweep(
                graph, args.ratio_grid, max_iters, args.num_attempts, SEED,
                SCHEDULE_BUILDERS[schedule_name],
            )
            elapsed = time.perf_counter() - t0
            print(f"\n[max_iters={max_iters} schedule={schedule_name}] sweep took {elapsed:.1f}s")
            print(f"{'r_start':>8s} {'r_end':>8s} {'avg_best':>9s} {'avg_time':>9s}")

            metric = (target - avg_best) if target is not None else avg_best
            metric_by_max_iters[max_iters][schedule_name] = metric

    metric_label = "avg optimality gap (known alpha - best found)" if target is not None else "avg best found"
    cmap = "viridis_r" if target is not None else "viridis"

    all_vals = np.concatenate([
        m.ravel() for per_schedule in metric_by_max_iters.values() for m in per_schedule.values()
    ])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    mid = (vmin + vmax) / 2

    n = len(args.ratio_grid)
    n_rows = len(args.max_iters_grid)
    n_cols = len(args.schedules)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 6.5 * n_rows), squeeze=False)

    im = None
    for row, max_iters in enumerate(args.max_iters_grid):
        for col, schedule_name in enumerate(args.schedules):
            ax = axes[row][col]
            metric = metric_by_max_iters[max_iters][schedule_name]
            im = ax.imshow(metric, origin="lower", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

            ax.set_xticks(range(n))
            ax.set_xticklabels([f"{r:g}" for r in args.ratio_grid], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels([f"{r:g}" for r in args.ratio_grid], fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("ratio_end (B/A at it=max_iters-1)")
            if col == 0:
                ax.set_ylabel(f"max_iters={max_iters}\nratio_start (B/A at it=0)")
            ax.set_title(schedule_name if row == 0 else "", fontsize=11)

            for i in range(n):
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.5))

            if not args.no_annotate:
                for i in range(n):
                    for j in range(n):
                        color = "white" if metric[i, j] < mid else "black"
                        ax.text(j, i, f"{metric[i, j]:.1f}", ha="center", va="center", color=color, fontsize=6)

    fig.suptitle(f"{name}: A/B ratio schedule sweep across budgets ({metric_label})")
    cbar = fig.colorbar(im, ax=axes, shrink=0.7)
    cbar.set_label(metric_label)

    schedules_tag = "-".join(args.schedules)
    out_path = os.path.join(base_dir, "results", f"ratio_schedule_sweep_by_budget_{name}_{schedules_tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()