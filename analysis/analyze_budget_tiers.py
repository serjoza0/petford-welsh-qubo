"""Summarize benchmark_budget_tiers.py results per instance set.

Reads the results CSV (default results/budget_tiers_results.csv), takes the
best size found for each instance over all rows, and compares it to the
instance's upper bound. Instances are grouped by the folder they live in
under instances/ (dimacs, bhoslib, ...). Instances with no known upper bound
are ignored. For each set it counts the instances whose best result is
optimal (gap 0), and then cumulative counts (so each column includes the
optimal ones) that depend on GAP_MODE (or --mode), where
gap = upper_bound - best:

  percent   gap <= p% of the upper bound, for p in 1, 5, 10, 25, 50. Unfair to
            instances with a small optimum, where one vertex is already a
            large percentage.
  absolute  gap <= k vertices, for k in 1, 2, 3, 5, 10. Unfair to instances
            with a large optimum, where a gap of 10 is a tiny fraction.
  hybrid    the percent columns, except that being off by at most
            HYBRID_FLOOR vertices always counts. Fixes the small-optimum
            problem without hurting the large ones.

With AS_LATEX = True (or --latex) two LaTeX tables are written to report/
instead of printing the summary:

  budget_tiers_table.tex          one row per instance, in the same longtable
                                  style as benchmark_run_instance_cli.py, but
                                  one table (own caption and label
                                  tab:results_budget_tiers_<set>) for each
                                  instance set, with columns:
                                  n, m, best, known alpha (the upper bound),
                                  num_iters (iterations per attempt at the
                                  smallest tier that found that best), total
                                  solver time over all tiers, gap, relative gap;
  budget_tiers_summary_table.tex  the per-set summary described above.
"""
import os
import re
import csv
import argparse

from benchmark_common import *

AS_LATEX = False
GAP_MODE = "hybrid"
PERCENTS = [1, 2, 5, 10, 25]
ABSOLUTES = [1, 2, 3, 5, 10]
HYBRID_FLOOR = 1
TABLE_CAPTION = ("Rezultati izvajanja algoritma na instancah iz množice {set} z naraščajočim proračunom iteracij. "
                 "Za vsak primer so prikazani najboljša najdena velikost neodvisne množice (best), "
                 "znana vrednost (known alpha), "
                 "najmanjše število iteracij na poskus, pri katerem je bila najboljša velikost dosežena (num\\_iters), "
                 "in skupni čas računanja po vseh stopnjah proračuna (time\\_sec). "
                 "Uporabljeni so bili $A = 1.0$, $B \\in [1, 5.5]$ in baze $b \\in [1.5, 50]$.")
TABLE_LABEL = "tab:results_budget_tiers_{folder}"
TABLE_CONTINUED = "Nadaljevanje na naslednji strani"
TABLE_COLUMNS = {"upper_bound": "known alpha", "elapsed_sec": "time_sec", "relative_gap": "rel_gap"}
SET_NAMES = {"dimacs": "DIMACS", "bhoslib": "BHOSLIB", "PACE": "PACE 2019", "quantum": "Quantum",
             "EVIL": "EVIL", "coding_theory": "Coding theory"}


def best_per_instance(path):
    best = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("instance") or not row.get("best") or not row.get("upper_bound"):
                continue
            ub, b = int(row["upper_bound"]), int(row["best"])
            prev = best.get(row["instance"])
            best[row["instance"]] = (max(b, prev[0]) if prev else b, ub)
    return best


def instance_table(path, base_dir):
    per = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if any(row.get(k) in (None, "") for k in ("instance", "best", "factor", "max_iters", "n", "m", "elapsed_sec")):
                continue
            per.setdefault(row["instance"], []).append(row)

    rows = []
    for name in sorted(per, key=str.lower):
        rs = per[name]
        try:
            folder = os.path.basename(os.path.dirname(instance_path(name, base_dir)))
        except FileNotFoundError:
            continue
        best = max(int(r["best"]) for r in rs)
        first = min((r for r in rs if int(r["best"]) == best), key=lambda r: float(r["factor"]))
        ub = int(rs[0]["upper_bound"]) if rs[0]["upper_bound"] not in (None, "") else None
        rows.append({
            "set": folder,
            "instance": name,
            "n": int(first["n"]), "m": int(first["m"]),
            "best": best,
            "upper_bound": ub if ub is not None else "",
            "num_iters": int(first["max_iters"]),
            "elapsed_sec": sum(float(r["elapsed_sec"]) for r in rs),
            "gap": ub - best if ub is not None else "",
            "relative_gap": f"{(ub - best) / ub:.4f}" if ub is not None else "",
        })
    return rows


def longtable_for_set(df, folder):
    lines = df.to_latex(index=False, float_format="%.2f", escape=True).splitlines()
    spec = "".join("l" if c == "instance" else "r" for c in df.columns)
    top = lines.index(r"\toprule")
    mid = lines.index(r"\midrule")
    bottom = lines.index(r"\bottomrule")
    header = lines[top + 1]
    body = lines[mid + 1:bottom]
    caption = TABLE_CAPTION.replace("{set}", SET_NAMES.get(folder, folder).replace("_", r"\_"))
    label = TABLE_LABEL.replace("{folder}", folder)
    ncols = len(df.columns)
    return "\n".join([
        r"{\setlength{\LTcapwidth}{\textwidth}",
        r"\begin{longtable}{" + spec + "}",
        r"\caption{" + caption + r"} \label{" + label + r"} \\",
        r"\toprule", header, r"\midrule",
        r"\endfirsthead",
        r"\toprule", header, r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{" + TABLE_CONTINUED + r"} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        *body,
        r"\end{longtable}}",
    ])


def write_instance_latex(rows, tex_path):
    import pandas as pd
    df = pd.DataFrame(rows).rename(columns=TABLE_COLUMNS)
    order = {f: i for i, f in enumerate(INSTANCE_SUBDIRS)}
    tables = []
    for folder in sorted(df["set"].unique(), key=lambda f: order.get(f, len(order))):
        part = df[df["set"] == folder].drop(columns=["set"])
        tables.append(longtable_for_set(part, folder))
    with open(tex_path, "w") as f:
        f.write("\n\n".join(tables) + "\n")


def gap_columns(mode):
    if mode == "percent":
        return [(f"<= {p}%", lambda d, ub, p=p: d * 100 <= p * ub) for p in PERCENTS]
    if mode == "absolute":
        return [(f"<= {k}", lambda d, ub, k=k: d <= k) for k in ABSOLUTES]
    if mode == "hybrid":
        return [(f"<= {p}%*", lambda d, ub, p=p: d <= HYBRID_FLOOR or d * 100 <= p * ub)
                for p in PERCENTS]
    raise ValueError(f"unknown mode {mode!r}")


def set_stats(table_rows):
    groups = {}
    for r in table_rows:
        groups.setdefault(r["set"], []).append(r)
        groups.setdefault("all", []).append(r)
    stats = {}
    for folder, rs in groups.items():
        times = [r["elapsed_sec"] for r in rs]
        rels = [float(r["relative_gap"]) for r in rs if r["relative_gap"] != ""]
        stats[folder] = (f"{sum(times) / len(times):.2f}",
                         f"{sum(rels) / len(rels):.4f}" if rels else "")
    return stats


def summarize(best, base_dir, columns, stats=None):
    groups = {}
    for name, (b, ub) in best.items():
        try:
            folder = os.path.basename(os.path.dirname(instance_path(name, base_dir)))
        except FileNotFoundError:
            continue
        groups.setdefault(folder, []).append((ub - b, ub))

    order = {f: i for i, f in enumerate(INSTANCE_SUBDIRS)}
    rows = []
    for folder in sorted(groups, key=lambda f: order.get(f, len(order))):
        rows.append(count_row(folder, groups[folder], columns) + list((stats or {}).get(folder, ("", ""))))
    rows.append(count_row("all", [g for gs in groups.values() for g in gs], columns)
                + list((stats or {}).get("all", ("", ""))))
    return rows


def count_row(label, gaps, columns):
    return [label, len(gaps), sum(1 for d, ub in gaps if d <= 0)] + \
           [sum(1 for d, ub in gaps if test(d, ub)) for _, test in columns]


def format_plain(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    lines = ["  ".join(str(v).ljust(w) if i == 0 else str(v).rjust(w)
                       for i, (v, w) in enumerate(zip(r, widths))) for r in [header] + rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def format_latex(rows, header):
    esc = lambda s: (str(s).replace("_", r"\_").replace("%", r"\%")
                     .replace("<=", r"$\leq$").replace("*", r"\textsuperscript{*}"))
    body = [" & ".join(esc(v) for v in r) + r" \\" for r in rows]
    return "\n".join([
        r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
        r"\toprule",
        " & ".join(esc(h) for h in header) + r" \\",
        r"\midrule",
        *body[:-1],
        r"\midrule",
        body[-1],
        r"\bottomrule",
        r"\end{tabular}",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                         help="results CSV; default results/budget_tiers_results.csv")
    parser.add_argument("--mode", choices=["percent", "absolute", "hybrid"], default=None,
                         help="how gaps are bucketed; default is GAP_MODE at the top of the file")
    parser.add_argument("--latex", action="store_true",
                         help="write a LaTeX table instead of printing (same as AS_LATEX = True)")
    args = parser.parse_args()

    base_dir = os.getcwd()
    path = args.input or results_path("budget_tiers_results.csv", base_dir)
    mode = args.mode or GAP_MODE
    columns = gap_columns(mode)
    table_rows = instance_table(path, base_dir)
    rows = summarize(best_per_instance(path), base_dir, columns, set_stats(table_rows))
    header = ["set", "instances", "optimal"] + [h for h, _ in columns] + ["avg time_sec", "mean rel_gap"]
    note = (f"* or off by at most {HYBRID_FLOOR} vertex" + ("" if HYBRID_FLOOR == 1 else "es")
            if mode == "hybrid" else None)

    if AS_LATEX or args.latex:
        report_dir = os.path.join(base_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        table_path = os.path.join(report_dir, "budget_tiers_table.tex")
        write_instance_latex(table_rows, table_path)
        print(f"Wrote LaTeX table to {table_path}")
        tex_path = os.path.join(report_dir, "budget_tiers_summary_table.tex")
        with open(tex_path, "w") as f:
            f.write(format_latex(rows, header) + "\n")
            if note:
                f.write("% " + note + "\n")
        print(f"Wrote LaTeX table to {tex_path}")
    else:
        print(format_plain(rows, header))
        if note:
            print(note)


if __name__ == "__main__":
    main()
