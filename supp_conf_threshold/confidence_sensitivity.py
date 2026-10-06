import os
from typing import List, Optional

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from supp_conf_threshold.run_adaptive_support_pipeline import discover_datasets, normalize_onehot, DATA_ROOT

FIXED_SUPPORT = 0.2
CONFIDENCE_GRID = [0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
LOW_MEMORY = True


def evaluate_confidence_grid(
    dataset_names: Optional[List[str]] = None,
    data_root: str = DATA_ROOT,
    support: float = FIXED_SUPPORT,
    confidence_grid: List[float] = None,
):
    if confidence_grid is None:
        confidence_grid = CONFIDENCE_GRID

    if dataset_names is None:
        dataset_names = discover_datasets(data_root)
        if not dataset_names:
            print(f"No dataset found in '{data_root}/*/*_apriori.csv'.")
            return pd.DataFrame(), pd.DataFrame()

    print(f"Datasets found ({len(dataset_names)}): {dataset_names}")
    print(f"Fixed min_support = {support}")
    print(f"Confidence grid tested = {confidence_grid}\n")

    all_results = []

    for name in dataset_names:
        csv_path = os.path.join(data_root, name, f"{name}_apriori.csv")

        if not os.path.exists(csv_path):
            print(f"[SKIP] {name}: file not found ({csv_path})")
            continue

        try:
            df_raw = pd.read_csv(csv_path)
            df_bool = normalize_onehot(df_raw)
        except Exception as e:
            print(f"[NORMALIZATION ERROR] {name}: {e}")
            continue

        try:
            frequent_itemsets = apriori(
                df_bool, min_support=support, use_colnames=True,
                low_memory=LOW_MEMORY,
            )
        except MemoryError as e:
            print(f"[MEMORY ERROR] {name}: ({e}) ")
            continue
        except Exception as e:
            print(f"[APRIORI ERROR] {name}: {e}")
            continue

        if frequent_itemsets.empty:
            print(f"[WARNING] {name}: no frequent itemset with support={support} "
                  f"-> unable to generate rules, dataset skipped")
            continue

        for conf in confidence_grid:
            try:
                rules = association_rules(
                    frequent_itemsets, metric="confidence", min_threshold=conf
                )
            except Exception:
                rules = pd.DataFrame()

            n_rules = len(rules)
            n_lift_gt1 = int((rules["lift"] > 1).sum()) if n_rules else 0
            pct_lift_gt1 = round(n_lift_gt1 / n_rules * 100, 2) if n_rules else 0.0

            all_results.append({
                "dataset": name,
                "support": support,
                "confidence": conf,
                "n_total_rules": n_rules,
                "n_rules_lift_gt1": n_lift_gt1,
                "pct_rules_lift_gt1": pct_lift_gt1,
            })

    results_df = pd.DataFrame(all_results)


    min_rule_count = 4
    best_rows = []
    if not results_df.empty:
        for name, group in results_df.groupby("dataset"):
            eligible = group[group["n_total_rules"] >= min_rule_count]
            if eligible.empty:

                best_rows.append({
                    "dataset": name,
                    "recommended_confidence": None,
                    "n_total_rules": None,
                    "pct_rules_lift_gt1": None,
                    "note": f"no confidence value produces >= {min_rule_count} rules, "
                            f"consider a lower support for this dataset",
                })
                continue
            best_row = eligible.loc[eligible["pct_rules_lift_gt1"].idxmax()]
            best_rows.append({
                "dataset": name,
                "recommended_confidence": best_row["confidence"],
                "n_total_rules": best_row["n_total_rules"],
                "pct_rules_lift_gt1": best_row["pct_rules_lift_gt1"],
                "note": "",
            })
    best_df = pd.DataFrame(best_rows)

    return results_df, best_df


if __name__ == "__main__":
    results_df, best_df = evaluate_confidence_grid()

    if not results_df.empty:
        print("\nFull table (dataset x confidence):\n")
        print(results_df.to_string(index=False))
        results_df.to_csv("confidence_sensitivity_full.csv", index=False)
        print("\nSaved to: confidence_sensitivity_full.csv")

    if not best_df.empty:
        print("\n\nBest confidence value per dataset:\n")
        print(best_df.to_string(index=False))
        best_df.to_csv("confidence_sensitivity_best.csv", index=False)
        print("\nSaved to: confidence_sensitivity_best.csv")
