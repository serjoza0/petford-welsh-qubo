import os
import sys
import argparse
from time import perf_counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm, Normalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scripts import *

from dwave.samplers import SimulatedAnnealingSampler
from qpu_comparison import *

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

DEFAULT_MAX_ITERS = [5000, 20000, 100000, 500000, 2000000, 10000000, 50000000, 100000000]
A = 1.0
B = 10.0
b = 18.0
SEED = 0
SA_BETA = 0.5
BOX_WIDTH_FRAC = 0.15  # box width as a fraction of its x-position, so boxes look even on the log x-axis


def run_pw_sweep(graph, max_iters_list, num_attempts, seed, solver):
    """(max_iters, best_value, elapsed) for every attempt at every max_iters level."""
    points = []
    for max_iters in max_iters_list:
        result = run_multi_start(
            graph, solver=solver, A=A, B=B, b=b,
            max_iters=max_iters, num_attempts=num_attempts, seed=seed,
        )
        points.extend((max_iters, a.best_value, a.elapsed) for a in result.attempts)
    return points


def run_sa_attempts(nx_graph, num_attempts, beta, sampler):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="C125.9")
    parser.add_argument("--max-iters", type=int, nargs="+", default=DEFAULT_MAX_ITERS)
    parser.add_argument("--num-attempts", type=int, default=20)
    parser.add_argument("--solver", choices=["python", "jit"], default="jit")
    parser.add_argument("--skip-sa", action="store_true")
    args = parser.parse_args()

    base_dir = os.getcwd()
    name = args.instance
    target = KNOWN_ALPHA.get(name)
    path = os.path.join(base_dir, "instances", "stable_set", f"{name}_stable_set_edge_list.txt")
    graph = CSRGraph.from_edge_list_file(path, name=name)
    if args.solver == "jit":
        petford_welsh_jit(graph, A=A, B=B, b=b, max_iters=10, rng=np.random.default_rng(1))

    pw_points = run_pw_sweep(graph, args.max_iters, args.num_attempts, SEED, args.solver)
    print(f"--- {name} (n={graph.n}, m={graph.m}) known alpha={target} ---")
    print(f"{'max_iters':>10s} {'avg_time':>10s} {'avg_best':>9s} {'avg_gap':>8s}")
    group_stats = []  # (max_iters, avg_time, avg_best, avg_gap_or_None)
    for max_iters in args.max_iters:
        vals = [(v, t) for (mi, v, t) in pw_points if mi == max_iters]
        best_vals = [v for v, t in vals]
        times = [t for v, t in vals]
        avg_time, avg_best = float(np.mean(times)), float(np.mean(best_vals))
        avg_gap = (target - avg_best) if target is not None else None
        group_stats.append((max_iters, avg_time, avg_best, avg_gap))
        gap_str = f"{avg_gap:8.2f}" if avg_gap is not None else "     n/a"
        print(f"{max_iters:10d} {avg_time:10.4f} {avg_best:9.2f} {gap_str}")

    sa_points = []
    if not args.skip_sa:
        sampler = SimulatedAnnealingSampler()
        n, edges = read_edge_list(path)
        nx_graph = create_networkx_graph(n, edges)
        sa_points = run_sa_attempts(nx_graph, args.num_attempts, SA_BETA, sampler)
        sa_times = [t for _, t in sa_points]
        sa_bests = [v for v, _ in sa_points]
        sa_avg_gap = (target - float(np.mean(sa_bests))) if target is not None else float("nan")
        print(f"\nSA reference: avg_time={np.mean(sa_times):.4f}s avg_best={np.mean(sa_bests):.2f} avg_gap={sa_avg_gap:.2f}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    n_levels = len(args.max_iters)
    lo, hi = min(args.max_iters), max(args.max_iters)
    norm = LogNorm(vmin=lo, vmax=hi) if hi > lo else Normalize(vmin=lo - 1, vmax=lo + 1)
    color_for = {mi: cmap(norm(mi)) for mi in args.max_iters}

    def y_of(v):
        return (target - v) if target is not None else v

    # one boxplot per max_iters level instead of raw per-attempt scatter,
    # positioned at that level's average time. Widths scale with position
    # (not fixed) so the boxes look even on the log-scaled x-axis.
    box_data, positions, widths, box_colors = [], [], [], []
    for max_iters, avg_time, avg_best, avg_gap in group_stats:
        ys = [y_of(v) for (mi, v, t) in pw_points if mi == max_iters]
        box_data.append(ys)
        positions.append(avg_time)
        widths.append(avg_time * BOX_WIDTH_FRAC)
        box_colors.append(color_for[max_iters])

    bp = ax.boxplot(
        box_data, positions=positions, widths=widths, patch_artist=True,
        manage_ticks=True, zorder=4,
        medianprops=dict(color="black", linewidth=1.3),
        meanprops=dict(color="black", linewidth=1.3, linestyle="--"),
        whiskerprops=dict(color="black"), capprops=dict(color="black"),
        flierprops=dict(marker="o", markersize=4, markeredgecolor="black", alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")

    if sa_points:
        for v, t in sa_points:
            ax.scatter(t, y_of(v), color="crimson", marker="^", alpha=0.6, s=30)

    ax.set_xlabel("time per attempt (s)")
    ax.set_ylabel("optimality gap (known alpha - best found)" if target is not None else "best found")
    ax.set_title(f"{name}: PW max_iters sweep vs. time/quality trend" + (" (SA reference in red)" if sa_points else ""))
    ax.set_xscale("log")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("max_iters")

    handles = [
    ]
    if sa_points:
        handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="crimson", markersize=8, label="SA (reference)"))
    ax.legend(handles=handles, loc="best", fontsize=8)

    out_path = os.path.join(base_dir, "results", f"time_to_best_vs_gap_{name}_maxiters_sweep.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path} ({len(pw_points)} PW points across {n_levels} max_iters levels"
          + (f", {len(sa_points)} SA points" if sa_points else "") + ")")


if __name__ == "__main__":
    main()