"""Benchmark scripts/run_instance.py, unmodified, across every instance.

For each instance, invokes ``python3 scripts/run_instance.py <path>`` as a
subprocess with no extra flags, so run_instance.py's own defaults decide
effort (its DEFAULT_NUM_ATTEMPTS / DEFAULT_TIME_LIMIT calibration). Pass
--time-limit to forward run_instance.py's own --time-limit (total wall-clock
seconds across all attempts for that instance) instead of its default.

run_instance.py is never given --known-alpha, so upper_bound is looked up
separately from instances/mcp_best_known_solutions.csv (matched to project
instance names the same way KNOWN_ALPHA is, but keeping every matched row's
upper_bound -- including the ones not proven optimal, unlike KNOWN_ALPHA).

Writes one row per instance to results/run_instance_cli_results.csv, prints
a summary table, and (with --latex) also writes a LaTeX table to
report/run_instance_cli_table.tex.
"""
import os
import re
import csv
import sys
import argparse
import subprocess

from benchmark_common import *

BEST_LINE_RE = re.compile(r"best=(\d+)\s+mean=([\d.]+)\s+std=([\d.]+)")
TIME_LINE_RE = re.compile(r"in ([\d.]+)s\s*\(actual")
HEADER_LINE_RE = re.compile(r"^(?P<name>\S+): n=(?P<n>\d+)\s+m=(?P<m>\d+)", re.MULTILINE)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_upper_bounds(csv_path, project_names):
    """Match every CSV row to a project instance name, by normalized name.

    Unlike KNOWN_ALPHA, this keeps every matched row's upper_bound, whether
    or not it equals best_known (i.e. including unproven bounds).
    """
    by_norm = {norm(name): name for name in project_names}
    upper_bound = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            candidates = [row["instance"], os.path.splitext(row["file"])[0]]
            for cand in candidates:
                key = norm(cand)
                if key in by_norm:
                    upper_bound[by_norm[key]] = int(row["upper_bound"])
                    break
    return upper_bound


def run_one(run_instance_py, path, base_dir, time_limit=None):
    cmd = [sys.executable, run_instance_py, path]
    if time_limit is not None:
        cmd += ["--time-limit", str(time_limit)]
    proc = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True)

    if proc.returncode != 0:
        return {"status": "error", "stderr": proc.stderr.strip().splitlines()[-1] if proc.stderr else ""}

    out = proc.stdout
    header = HEADER_LINE_RE.search(out)
    best_match = BEST_LINE_RE.search(out)
    time_match = TIME_LINE_RE.search(out)
    if not (header and best_match):
        return {"status": "unparseable"}

    return {
        "status": "ok",
        "n": int(header["n"]), "m": int(header["m"]),
        "best": int(best_match.group(1)),
        "mean": float(best_match.group(2)),
        "std": float(best_match.group(3)),
        "elapsed_sec": float(time_match.group(1)) if time_match else None,
    }


def main():
    parser = argparse.ArgumentParser()
    add_instances_arg(parser)
    parser.add_argument("--time-limit", type=float, default=None,
                         help="forwarded to run_instance.py's --time-limit (total wall-clock seconds "
                              "per instance, across all its attempts); default is run_instance.py's own")
    parser.add_argument("--latex", action="store_true", help="also save a LaTeX table of results")
    args = parser.parse_args()

    base_dir = os.getcwd()
    run_instance_py = os.path.join(base_dir, "scripts", "run_instance.py")
    names = args.instances or all_instance_names(base_dir)

    upper_bound = load_upper_bounds(
        os.path.join(base_dir, "instances", "mcp_best_known_solutions.csv"), all_instance_names(base_dir)
    )

    rows = []
    for idx, name in enumerate(names, 1):
        path = instance_path(name, base_dir)
        print(f"[{idx}/{len(names)}] {name} ... ", end="", flush=True)
        outcome = run_one(run_instance_py, path, base_dir, args.time_limit)

        ub = upper_bound.get(name)
        row = {"instance": name, "upper_bound": ub if ub is not None else "", "status": outcome["status"]}
        if outcome["status"] == "ok":
            gap = (ub - outcome["best"]) if ub is not None else None
            relative_gap = (gap / ub) if gap is not None else None
            row.update({
                "n": outcome["n"], "m": outcome["m"],
                "best": outcome["best"], "mean": outcome["mean"], "std": outcome["std"],
                "elapsed_sec": outcome["elapsed_sec"],
                "gap": gap if gap is not None else "",
                "relative_gap": f"{relative_gap:.4f}" if relative_gap is not None else "",
            })
            print(f"best={outcome['best']:4d} upper_bound={ub if ub is not None else '?':>4} "
                  f"gap={row['gap']} time={outcome['elapsed_sec']:.2f}s")
        else:
            row.update({"n": "", "m": "", "best": "", "mean": "", "std": "", "elapsed_sec": "", "gap": "", "relative_gap": ""})
            print(outcome["status"] + (f" ({outcome.get('stderr', '')})" if outcome.get("stderr") else ""))
        rows.append(row)

    csv_path = results_path("run_instance_cli_results.csv", base_dir)
    write_csv(rows, csv_path)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    ok_rows = [r for r in rows if r["status"] == "ok"]
    with_gap = [r for r in ok_rows if r["gap"] != ""]
    print(f"\n{len(ok_rows)}/{len(rows)} instances solved, {len(with_gap)} with a known upper_bound")
    if with_gap:
        avg_relative_gap = sum(float(r["relative_gap"]) for r in with_gap) / len(with_gap)
        print(f"average relative gap over those {len(with_gap)}: {avg_relative_gap:.4f}")

    if args.latex:
        import pandas as pd
        df = pd.DataFrame(rows)
        df = df.drop(columns=["std", "status"], errors="ignore")
        latex_table = df.to_latex(index=False, float_format="%.2f", escape=True, longtable=True,
                                  caption="Rezultati izvajanja algoritma na testnih instancah. Za vse primere so bili uporabljeni parametri $A = 1.0$, $B = 2.0$, baza $b = 8$. Število ponovitev in število iteracij pa sta bili konstantni.",
                                  label="tab:results_solver"
        )
        report_dir = os.path.join(base_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        tex_path = os.path.join(report_dir, "run_instance_cli_table.tex")
        with open(tex_path, "w") as f:
            f.write(latex_table)
        print(f"Wrote LaTeX table to {tex_path}")


if __name__ == "__main__":
    main()
