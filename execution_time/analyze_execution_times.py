"""
analyze_execution_times.py
-----------------------------
Automatically extracts the execution times for EVERY dataset and EVERY
variant found (no list to specify by hand), keeps only the LATEST run
when there is more than one for the same (dataset, variant) combination,
and generates:
  - a full CSV with every run found (for transparency/debugging)
  - a filtered CSV with only the latest run per combination
  - a bar chart of the relative overhead with respect to a baseline

The "latest run" is determined from the run_id (e.g. "run_2026_08_25_20_31_11"),
compared as a string: since the format is zero-padded YYYY_MM_DD_HH_MM_SS,
lexicographic order matches chronological order.

Usage:
    python analyze_execution_times.py results
    python analyze_execution_times.py results --time-col total_execution_time_sec
    python analyze_execution_times.py results --baseline B
"""

from __future__ import annotations
import argparse
import ast
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


VARIANT_MAP = {
    "baseline": "B",
    "CAR": "N",
    "CAR_Accuracy": "NA",
    "CAR_TopK": "NK",
    "Split_Accuracy": "NSA",
    "Split_TopK": "NSK",
    "Split_Lift_Accuracy": "NSAL",
    "Split_Lift_TopK": "NSKL",
}

TIME_COLUMNS = ["grid_search_time_sec", "final_training_time_sec", "total_execution_time_sec"]

METRICS_DICT_RE = re.compile(r"INFO - (\{.*'cv_mean_score'.*\})\s*$")
RESULTS_DIR_RE = re.compile(r"Results directory:\s*(\S+)")

def parse_log_file(path: Path, results_root: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")

    metrics_match = METRICS_DICT_RE.search(text)
    if not metrics_match:
        return None

    try:
        metrics = ast.literal_eval(metrics_match.group(1))
    except (ValueError, SyntaxError):
        return None

    dir_match = RESULTS_DIR_RE.search(text)
    if dir_match:
        rel_parts = Path(dir_match.group(1)).parts
        try:
            root_idx = rel_parts.index(results_root.name)
            dataset = rel_parts[root_idx + 1]
            variant_raw = rel_parts[root_idx + 2]
            run_id = rel_parts[root_idx + 3] if len(rel_parts) > root_idx + 3 else None
        except (ValueError, IndexError):
            rel = path.relative_to(results_root)
            dataset, variant_raw = rel.parts[0], rel.parts[1]
            run_id = rel.parts[2] if len(rel.parts) > 2 else None
    else:
        rel = path.relative_to(results_root)
        dataset, variant_raw = rel.parts[0], rel.parts[1]
        run_id = rel.parts[2] if len(rel.parts) > 2 else None

    return dict(
        dataset=dataset,
        variant_raw=variant_raw,
        run_id=run_id,
        log_path=str(path),
        cv_mean_score=metrics.get("cv_mean_score"),
        cv_std_score=metrics.get("cv_std_score"),
        grid_search_time_sec=metrics.get("grid_search_time_sec"),
        final_training_time_sec=metrics.get("final_training_time_sec"),
        total_execution_time_sec=metrics.get("total_execution_time_sec"),
    )


def scan_all_logs(results_root: Path, pattern: str = "**/*.log") -> pd.DataFrame:
    log_files = sorted(results_root.glob(pattern))
    records = [r for r in (parse_log_file(p, results_root) for p in log_files) if r is not None]

    if not records:
        raise RuntimeError(f"No valid logs found in {results_root} with pattern '{pattern}'")

    df = pd.DataFrame(records)
    df["variant"] = df["variant_raw"].map(VARIANT_MAP)

    unmapped = df[df["variant"].isna()]["variant_raw"].unique()
    if len(unmapped) > 0:
        print(f"WARNING: folders not mapped in VARIANT_MAP, excluded from the analysis: {list(unmapped)}")
        df = df[df["variant"].notna()]

    return df



def keep_latest_run(df: pd.DataFrame) -> pd.DataFrame:
    """For each (dataset, variant), keeps only the row with the most recent
    run_id (lexicographic comparison, valid for the run_YYYY_MM_DD_HH_MM_SS format)."""
    df = df.copy()
    df["run_id"] = df["run_id"].fillna("")
    idx = df.groupby(["dataset", "variant"])["run_id"].idxmax()
    latest = df.loc[idx].reset_index(drop=True)
    return latest


def compute_relative_overhead(df_latest: pd.DataFrame, baseline: str, time_col: str) -> pd.DataFrame:
    if baseline not in df_latest["variant"].unique():
        raise ValueError(
            f"Baseline '{baseline}' not found among the available variants: "
            f"{sorted(df_latest['variant'].unique())}"
        )

    baseline_times = (
        df_latest[df_latest["variant"] == baseline]
        .set_index("dataset")[time_col]
        .rename("baseline_time_sec")
    )

    merged = df_latest.set_index("dataset").join(baseline_times).reset_index()

    missing = merged[merged["baseline_time_sec"].isna()]["dataset"].unique()
    if len(missing) > 0:
        print(f"WARNING: baseline '{baseline}' missing for datasets {list(missing)}, "
              f"excluded from the overhead calculation for those datasets.")
    merged = merged.dropna(subset=["baseline_time_sec"])

    merged["relative_overhead"] = merged[time_col] / merged["baseline_time_sec"]

    summary = (
        merged.groupby("variant")["relative_overhead"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean")
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", help="Root folder of the logs (e.g. results)")
    parser.add_argument("--pattern", default="**/*.log")
    parser.add_argument("--time-col", default="final_training_time_sec", choices=TIME_COLUMNS)
    parser.add_argument("--baseline", default="N", help="Reference variant (default: N)")
    parser.add_argument("--output-dir", default="execution_time", help="Folder where CSVs are saved (default: execution_time)")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = scan_all_logs(results_root, pattern=args.pattern)
    df_all.to_csv(output_dir / "execution_times_all_runs.csv", index=False)
    print(f"Found {len(df_all)} valid logs, {df_all['dataset'].nunique()} datasets, "
          f"{df_all['variant'].nunique()} variants.")

    df_latest = keep_latest_run(df_all)
    df_latest.to_csv(output_dir / "execution_times_latest_run.csv", index=False)

    n_datasets = df_latest["dataset"].nunique()
    n_variants = df_latest["variant"].nunique()
    print(f"After 'latest run' selection: {len(df_latest)} rows "
          f"({n_datasets} datasets x {n_variants} variants expected: {n_datasets * n_variants}).")
    if len(df_latest) < n_datasets * n_variants:
        pivot = df_latest.pivot_table(index="dataset", columns="variant", values=args.time_col, aggfunc="first")
        missing_cells = pivot.isna().sum().sum()
        print(f"WARNING: {missing_cells} (dataset, variant) combinations missing. Details:")
        print(pivot.isna().to_string())

    summary = compute_relative_overhead(df_latest, baseline=args.baseline, time_col=args.time_col)
    print(f"\nRelative overhead per variant ({args.time_col}, baseline: {args.baseline})")
    print(summary.to_string(index=False))
    summary.to_csv(output_dir / "execution_times_overhead_summary.csv", index=False)


if __name__ == "__main__":
    main()
