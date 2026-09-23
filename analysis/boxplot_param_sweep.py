"""Boxplot the distribution of PW stable-set sizes across a 1-D parameter
sweep, on a single instance.

Sweeps either the PW base b (with A and B fixed), or the B/A ratio (with b
fixed and A fixed, B computed as A * ratio). For each value in the sweep,
runs num_attempts PW attempts and boxplots the resulting best-set sizes
side by side, with a horizontal dashed line at the instance's known alpha
(from KNOWN_ALPHA), if known. Saves the figure to
results/boxplot_<instance>_<sweep-over>.png.
"""
import os
import argparse
import matplotlib.pyplot as plt

from benchmark_common import *
from scripts import *

DEFAULT_BASE_GRID = [3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 75, 100]
DEFAULT_RATIO_GRID = [1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]


def run_sweep(graph, sweep_over, values, A, B, base, max_iters, num_attempts, seed, known_alpha):
    sizes_by_value = []
    for val in values:
        if sweep_over == "base":
            A_i, B_i, b_i = A, B, val
        else:
            A_i, b_i = A, base
            B_i = A * val
        result = run_multi_start(
            graph, A=A_i, B=B_i, b=b_i,
            max_iters=max_iters, num_attempts=num_attempts, seed=seed,
            known_alpha=known_alpha, show_progress=False,
        )
        sizes_by_value.append(result.sizes)
        print(f"  {sweep_over}={val:g}: best={result.best} mean={result.mean:.2f} std={result.std:.2f}")
    return sizes_by_value


FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 16


def plot_boxplot(values, sizes_by_value, known_alpha, xlabel, title, color, out_path):
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(values) + 3), 7.5))
    box = ax.boxplot(
        sizes_by_value, positions=range(len(values)), widths=0.65,
        patch_artist=True, showmeans=True,
        boxprops={"linewidth": 2},
        whiskerprops={"linewidth": 2},
        capprops={"linewidth": 2},
        medianprops={"linewidth": 2.5, "color": "black"},
        meanprops={"markersize": 10, "markeredgewidth": 1.5,
                   "markerfacecolor": "white", "markeredgecolor": "black"},
        flierprops={"markersize": 8, "markeredgewidth": 1.5},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor("black")

    if known_alpha is not None:
        ax.axhline(known_alpha, color="black", linestyle="--", linewidth=2.5,
                   label=f"known alpha = {known_alpha}")
        ax.legend(fontsize=LEGEND_SIZE, loc="best")

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([f"{v:g}" for v in values], rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, labelpad=12)
    ax.set_ylabel("stable set size", fontsize=LABEL_SIZE, labelpad=12)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=16)
    ax.grid(axis="y", alpha=0.4, linewidth=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    add_instance_arg(parser)
    parser.add_argument("--sweep-over", choices=["base", "ratio"], default="base",
                         help="'base' sweeps b with A,B fixed; 'ratio' sweeps B/A with A,b fixed")
    parser.add_argument("--values", type=float, nargs="+", default=None,
                         help="parameter values to sweep; default depends on --sweep-over")
    parser.add_argument("--A", type=float, default=DEFAULT_A)
    parser.add_argument("--B", type=float, default=DEFAULT_B,
                         help="fixed B, used when --sweep-over base")
    parser.add_argument("--base", type=float, default=DEFAULT_BASE,
                         help="fixed b, used when --sweep-over ratio")
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--num-attempts", type=int, default=DEFAULT_NUM_ATTEMPTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--color", default="tab:blue", help="matplotlib color for the boxplots")
    parser.add_argument("--output", default=None,
                         help="output PNG path; default results/boxplot_<instance>_<sweep-over>.png")
    args = parser.parse_args()

    base_dir = os.getcwd()
    graph = load_instance(args.instance, base_dir)
    known_alpha = KNOWN_ALPHA.get(args.instance)

    jit_warmup(graph)

    if args.sweep_over == "base":
        values = args.values or DEFAULT_BASE_GRID
        xlabel = f"base b  (A={args.A:g}, B={args.B:g})"
    else:
        values = args.values or DEFAULT_RATIO_GRID
        xlabel = f"B/A ratio  (A={args.A:g}, b={args.base:g})"

    print(f"--- {args.instance} (n={graph.n}, m={graph.m}) known_alpha={known_alpha} ---")
    print(f"sweeping {args.sweep_over} over {values}, {args.num_attempts} attempts each, "
          f"max_iters={args.max_iters}")

    sizes_by_value = run_sweep(
        graph, args.sweep_over, values, args.A, args.B, args.base,
        args.max_iters, args.num_attempts, args.seed, known_alpha,
    )

    out_path = args.output or results_path(
        f"boxplot_{args.instance}_{args.sweep_over}.png", base_dir
    )
    title = f"{args.instance}: stable set size vs {args.sweep_over}"
    out_path = plot_boxplot(values, sizes_by_value, known_alpha, xlabel, title, args.color, out_path)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
