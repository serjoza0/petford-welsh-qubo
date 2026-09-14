import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scripts import *
from benchmark_common import KNOWN_ALPHA, DEFAULT_A, DEFAULT_B, DEFAULT_BASE, DEFAULT_MAX_ITERS, DEFAULT_NUM_ATTEMPTS, DEFAULT_SEED

DEFAULT_RATIO_GRID = [0.5, 1, 1.5, 2, 3, 5, 8, 12, 20]

SCHEDULE_BUILDERS = {
    "geometric": lambda r_start, r_end, max_iters: np.geomspace(r_start, r_end, num=max_iters),
    "linear":    lambda r_start, r_end, max_iters: np.linspace(r_start, r_end, num=max_iters),
}


def run_attempts(graph, B_arr, max_iters, num_attempts, seed):
    vals = []
    for i in range(num_attempts):
        rng = np.random.default_rng(seed + i)
        _, best_value, _, _ = petford_welsh_jit(
            graph, A=DEFAULT_A, B=B_arr, b=DEFAULT_BASE, max_iters=max_iters, rng=rng,
        )
        vals.append(best_value)
    return vals


def run_sweep_rel_gap(graph, target, ratio_grid, max_iters, num_attempts, seed, build_schedule):
    ng = len(ratio_grid)
    rel_gap = np.zeros((ng, ng))
    for i, r_start in enumerate(ratio_grid):
        for j, r_end in enumerate(ratio_grid):
            B_arr = build_schedule(r_start, r_end, max_iters)
            vals = run_attempts(graph, B_arr, max_iters, num_attempts, seed)
            avg_best = float(np.mean(vals))
            rel_gap[i, j] = (target - avg_best) / target
    return rel_gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=list(KNOWN_ALPHA))
    parser.add_argument("--ratio-grid", type=float, nargs="+", default=DEFAULT_RATIO_GRID)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--num-attempts", type=int, default=DEFAULT_NUM_ATTEMPTS)
    parser.add_argument("--schedules", choices=list(SCHEDULE_BUILDERS), nargs="+",
                         default=["geometric", "linear"])
    parser.add_argument("--no-annotate", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    unknown = [name for name in args.instances if name not in KNOWN_ALPHA]
    if unknown:
        raise ValueError(f"no known alpha for: {unknown}")

    n_cells = len(args.ratio_grid) ** 2
    total_runs = n_cells * len(args.schedules) * len(args.instances) * args.num_attempts
    print(f"instances={len(args.instances)}  ratio_grid={len(args.ratio_grid)}x{len(args.ratio_grid)}  "
          f"schedules={args.schedules}  num_attempts={args.num_attempts}  max_iters={args.max_iters}")
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

    rel_gap_sums = {s: np.zeros((len(args.ratio_grid), len(args.ratio_grid))) for s in args.schedules}
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
        for schedule_name in args.schedules:
            rel_gap = run_sweep_rel_gap(
                graph, target, args.ratio_grid, args.max_iters, args.num_attempts, DEFAULT_SEED,
                SCHEDULE_BUILDERS[schedule_name],
            )
            rel_gap_sums[schedule_name] += rel_gap
        n_ok += 1
        print(f" -- {time.perf_counter() - t0:.1f}s")

    total_elapsed = time.perf_counter() - t_start
    print(f"\ndone: {n_ok}/{len(args.instances)} instances in {total_elapsed:.1f}s")
    if n_ok == 0:
        print("no instances succeeded, nothing to plot.")
        return

    avg_rel_gap = {s: rel_gap_sums[s] / n_ok for s in args.schedules}

    print(f"\nmean rel_gap over {n_ok} instances:")
    for schedule_name in args.schedules:
        print(f"\n[{schedule_name}]")
        print(f"{'r_start':>8s} {'r_end':>8s} {'mean_rel_gap':>13s}")
        for i, r_start in enumerate(args.ratio_grid):
            for j, r_end in enumerate(args.ratio_grid):
                print(f"{r_start:8.2f} {r_end:8.2f} {avg_rel_gap[schedule_name][i, j]:13.4f}")

    all_vals = np.concatenate([m.ravel() for m in avg_rel_gap.values()])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    mid = (vmin + vmax) / 2

    n = len(args.ratio_grid)
    ns = len(args.schedules)
    fig, axes = plt.subplots(1, ns, figsize=(7.5 * ns, 7.0), squeeze=False)
    axes = axes[0]

    im = None
    for ax, schedule_name in zip(axes, args.schedules):
        metric = avg_rel_gap[schedule_name]
        im = ax.imshow(metric, origin="lower", cmap="viridis_r", aspect="equal", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{r:g}" for r in args.ratio_grid], rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{r:g}" for r in args.ratio_grid])
        ax.set_xlabel("ratio_end (B/A at it=max_iters-1)")
        ax.set_ylabel("ratio_start (B/A at it=0)")
        ax.set_title(schedule_name)

        for i in range(n):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))

        if not args.no_annotate:
            for i in range(n):
                for j in range(n):
                    color = "white" if metric[i, j] < mid else "black"
                    ax.text(j, i, f"{metric[i, j]:.3f}", ha="center", va="center", color=color, fontsize=7)

    fig.suptitle(f"mean rel. gap over {n_ok} instances -- A/B ratio schedule, averaged")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("mean rel. gap ((alpha - best)/alpha)")

    schedules_tag = "-".join(args.schedules)
    out_path = os.path.join(base_dir, "results", f"ratio_schedule_sweep_all_instances_{schedules_tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()