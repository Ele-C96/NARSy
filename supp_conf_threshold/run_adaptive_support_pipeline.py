"""
Pipeline: computes the adaptive minimum support (Hikmawati et al., 2021)
over a collection of already-prepared one-hot datasets.

Expected file structure: data/{dataset}/{dataset}_apriori.csv

Each CSV is a one-hot dataset, but some encode presence/absence of an item as 0/1
and others as True/False: this script normalizes everything to boolean first.

Usage:
    python run_adaptive_support_pipeline.py
"""

import glob
import os
from typing import List, Optional

import pandas as pd

from supp_conf_threshold.adaptive_support import adaptive_min_support

DATA_ROOT = "data"
ITEM_THRESHOLD = 150

BOOL_STRINGS = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "1.0": True,
    "0.0": False,
}


def discover_datasets(data_root: str = DATA_ROOT) -> List[str]:
    """Returns the names of all data/{dataset}/ folders containing {dataset}_apriori.csv."""
    pattern = os.path.join(data_root, "*", "*_apriori.csv")
    datasets = []
    for path in sorted(glob.glob(pattern)):
        folder = os.path.basename(os.path.dirname(path))
        if os.path.basename(path) == f"{folder}_apriori.csv":
            datasets.append(folder)
    return datasets


def normalize_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts every column of a one-hot DataFrame to dtype bool.

    Accepted formats: native booleans, numeric 0/1, and strings such as
    "True"/"False" or "0"/"1" (which appear in mistyped CSVs).
    Raises ValueError if a column is not binary.
    """
    df = df.copy()

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_bool_dtype(series):
            continue

        if series.dtype == object:
            mapped = series.astype(str).str.strip().str.lower().map(BOOL_STRINGS)
            if not mapped.notna().all():
                raise ValueError(
                    f"Column '{col}' does not look binary (0/1/True/False): "
                    f"unique values found = {series.unique()[:5]}"
                )
            df[col] = mapped.astype(bool)

        elif pd.api.types.is_numeric_dtype(series):
            uniques = set(series.dropna().unique().tolist())
            if not uniques.issubset({0, 1}):
                raise ValueError(
                    f"Numeric column '{col}' is not binary: "
                    f"unique values found = {sorted(uniques)[:5]}"
                )
            df[col] = series.astype(bool)

    return df


def count_items(df_bool: pd.DataFrame) -> int:
    """Number of one-hot items, i.e. the number of columns of the normalized dataset."""
    return df_bool.shape[1]


def onehot_to_transactions(df_bool: pd.DataFrame) -> List[List[str]]:
    """Converts a boolean one-hot DataFrame into a list of transactions (active column names)."""
    return [row.index[row.values].tolist() for _, row in df_bool.iterrows()]


def run_pipeline(
    dataset_names: Optional[List[str]] = None,
    data_root: str = DATA_ROOT,
    item_threshold: int = ITEM_THRESHOLD,
    mode: str = "full_dataset",
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computes the adaptive minimum support for each dataset and returns a summary table.

    Parameters
    ----------
    mode : 'full_dataset' or 'train_only'
        'full_dataset': computes min_sup on the entire CSV. Only for quick
        exploration (order of magnitude): it leaks information, because the
        threshold also sees the test set. Not valid for the reported results.
        'train_only': holds out a test split and computes min_sup on the training
        part only. This is the reference for what a real cross-validation fold
        would see (in the actual training loop it must be recomputed per fold).
    test_size, random_state : used only when mode='train_only'.
    """
    if dataset_names is None:
        dataset_names = discover_datasets(data_root)
        if not dataset_names:
            print(f"No dataset found in '{data_root}/*/*_apriori.csv'.")
            return pd.DataFrame()

    print(f"Datasets found ({len(dataset_names)}): {dataset_names}\n")

    if mode == "full_dataset":
        print("!" * 70)
        print("'full_dataset' MODE: the threshold is computed on the entire CSV")
        print("!" * 70 + "\n")

    results = []
    over_threshold = []

    for name in dataset_names:
        csv_path = os.path.join(data_root, name, f"{name}_apriori.csv")

        if not os.path.exists(csv_path):
            print(f"[SKIP] {name}: file not found ({csv_path})")
            continue

        try:
            df_bool = normalize_onehot(pd.read_csv(csv_path))
            if mode == "train_only":
                df_bool = df_bool.sample(frac=1 - test_size, random_state=random_state)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results.append(
                {
                    "dataset": name,
                    "mode": mode,
                    "n_transactions": None,
                    "n_items": None,
                    "exceeds_150_items": None,
                    "density": None,
                    "adaptive_min_support": None,
                    "adaptive_min_support_pct": None,
                    "note": f"error: {e}",
                }
            )
            continue

        n_items = count_items(df_bool)
        exceeds = n_items > item_threshold
        if exceeds:
            over_threshold.append(name)

        transactions = onehot_to_transactions(df_bool)

        try:
            min_sup, density = adaptive_min_support(transactions, return_density=True)
            note = (
                f"WARNING: >{item_threshold} items, method not validated by the authors "
                "in this regime"
                if exceeds
                else ""
            )
        except Exception as e:
            min_sup, density = None, None
            note = f"error computing min_sup: {e}"

        results.append(
            {
                "dataset": name,
                "mode": mode,
                "n_transactions": df_bool.shape[0],
                "n_items": n_items,
                "exceeds_150_items": exceeds,
                "density": round(density, 3) if density is not None else None,
                "adaptive_min_support": min_sup,
                "adaptive_min_support_pct": f"{min_sup:.2%}" if min_sup is not None else None,
                "note": note,
            }
        )

    print("=" * 60)
    if over_threshold:
        print(
            f"Datasets that EXCEED {item_threshold} items "
            "(method not validated by the authors in this regime):"
        )
        for name in over_threshold:
            print(f"  - {name}")
    else:
        print(f"No dataset exceeds the {item_threshold}-item threshold.")
    print("=" * 60 + "\n")

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# QUICK TEST (exploration only, not for final results)")
    print("#" * 70)
    summary_full = run_pipeline(mode="full_dataset")

    if not summary_full.empty:
        print("\nTable (full_dataset):\n")
        print(summary_full.to_string(index=False))
        summary_full.to_csv("adaptive_support_summary_full.csv", index=False)
        print("\nSaved to: adaptive_support_summary_full.csv")