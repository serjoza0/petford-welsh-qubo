import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from benchmark_common import *
from scripts import *

DEFAULT_B_GRID = [1.5, 2, 3, 5, 8, 12, 20, 35, 50, 100, 200]

SCHEDULE_BUILDERS = {
    "geometric": lambda b_start, b_end, max_iters: np.geomspace(b_start, b_end, num=max_iters),
    "linear":    lambda b_start, b_end, max_iters: np.linspace(b_start, b_end, num=max_iters),
}

def run_sweep_rel_gap(graph, target, b_grid, max_iters, num_attempts, seed, solver, build_schedule):
    ng = len(b_grid)
    rel_gap = np.zeros((ng, ng))
    for i, b_start in enumerate(b_grid):
        for j, b_end in enumerate(b_grid):
            b_schedule = build_schedule(b_start, b_end, max_iters)
            result = run_multi_start(
                graph, solver=solver, A=DEFAULT_A, B=DEFAULT_B, b=b_schedule,
                max_iters=max_iters, num_attempts=num_attempts, seed=seed,
            )
            avg_best = float(np.mean([a.best_value for a in result.attempts]))
            rel_gap[i, j] = (target - avg_best) / target
    return rel_gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-grid", type=float, nargs="+", default=DEFAULT_B_GRID)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--num-attempts", type=int, default=DEFAULT_NUM_ATTEMPTS)
    parser.add_argument("--solver", choices=["python", "jit"], default="jit")
    parser.add_argument("--schedules", choices=list(SCHEDULE_BUILDERS), nargs="+",
                         default=["geometric", "linear"])
    args = parser.parse_args()

    if any(b <= 1.0 for b in args.b_grid):
        raise ValueError(f"all b values must be > 1 (got {args.b_grid}); b<=1 makes ln(b)<=0, i.e. non-finite/negative temperature")


    base_dir = os.getcwd()
    names = all_instance_names(base_dir)

    if args.solver == "jit":
        graph = load_instance(names[0],base_dir)
        jit_warmup(graph, A=DEFAULT_A, B=DEFAULT_B, b=DEFAULT_BASE, seed=1)

    rel_gap_sums = {s: np.zeros((len(args.b_grid), len(args.b_grid))) for s in args.schedules}
    n_ok = 0
    t_start = time.perf_counter()

    for name in tqdm(names, desc="Processing instances"): 
        target = KNOWN_ALPHA[name]
        path = os.path.join(base_dir, "instances", "stable_set", f"{name}_stable_set_edge_list.txt")
        try:
            graph = CSRGraph.from_edge_list_file(path, name=name)
        except FileNotFoundError:
            continue

        t0 = time.perf_counter()
        for schedule_name in args.schedules:
            rel_gap = run_sweep_rel_gap(
                graph, target, args.b_grid, args.max_iters, args.num_attempts, DEFAULT_SEED, args.solver,
                SCHEDULE_BUILDERS[schedule_name],
            )
            rel_gap_sums[schedule_name] += rel_gap
        n_ok += 1

    total_elapsed = time.perf_counter() - t_start
    if n_ok == 0:
        print("no instances succeeded, nothing to plot.")
        return

    avg_rel_gap = {s: rel_gap_sums[s] / n_ok for s in args.schedules}

    for schedule_name in args.schedules:
        print(f"\n[{schedule_name}]")
        print(f"{'b_start':>8s} {'b_end':>8s} {'mean_rel_gap':>13s}")
        for i, b_start in enumerate(args.b_grid):
            for j, b_end in enumerate(args.b_grid):
                print(f"{b_start:8.2f} {b_end:8.2f} {avg_rel_gap[schedule_name][i, j]:13.4f}")

    all_vals = np.concatenate([m.ravel() for m in avg_rel_gap.values()])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    mid = (vmin + vmax) / 2

    n = len(args.b_grid)
    ns = len(args.schedules)
    fig, axes = plt.subplots(1, ns, figsize=(7.5 * ns, 7.0), squeeze=False)
    axes = axes[0]

    im = None
    for ax, schedule_name in zip(axes, args.schedules):
        metric = avg_rel_gap[schedule_name]
        im = ax.imshow(metric, origin="lower", cmap="viridis_r", aspect="equal", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{b:g}" for b in args.b_grid], rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{b:g}" for b in args.b_grid])
        ax.set_xlabel("b_end (value at it=max_iters-1; higher = colder finish)")
        ax.set_ylabel("b_start (value at it=0; lower = hotter start)")
        ax.set_title(schedule_name)

        for i in range(n):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))
            for j in range(n):
                color = "white" if metric[i, j] < mid else "black"
                ax.text(j, i, f"{metric[i, j]:.3f}", ha="center", va="center", color=color, fontsize=7)

    norm_label = "alpha"
    fig.suptitle(f"mean rel. gap over {n_ok} instances -- (alpha - best)/{norm_label}, averaged")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label(f"mean rel. gap ((alpha - best)/{norm_label})")

    schedules_tag = "-".join(args.schedules)
    out_path = os.path.join(base_dir, "results", f"schedule_sweep_all_instances_{schedules_tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()