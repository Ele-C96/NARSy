"""
run_pipeline.py

End-to-end pipeline. For each dataset in config.DATASETS:
    1. load the raw csv
    2. clean the headers and drop the ID columns
    3. categorize/discretize the features (src.preprocessing.categorize)
    4. one-hot encoding
    5. mine association rules, both global and per target class
    6. save to disk: preprocessed dataset, metadata, rules


"""

import os
import json

import pandas as pd

from dataset_registry import DATASETS, MIN_SUPPORT, MIN_CONFIDENCE
from src.preprocessing import (
    clean_header,
    drop_id_columns,
    split_composite_columns,
    categorize,
    one_hot_encode,
)
from src.mining import mine_rules, mine_rules_per_class


def _normalize_column_name(name: str) -> str:
    """Applies the same normalization as clean_header to a single column name
    (needed to match the config's target to the name after header cleaning)."""
    return (
        pd.Series([name])
        .str.strip()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
        .iloc[0]
    )


def process_dataset(name: str, cfg: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Processing dataset: {name}")
    print("=" * 60)

    data_dir = f"data/{name}"
    rules_dir = f"rules/{name}"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs("metadata", exist_ok=True)
    os.makedirs(rules_dir, exist_ok=True)

    raw_path = f"{data_dir}/{cfg['file']}"
    pre_path = f"{data_dir}/{name}_pre.csv"
    apriori_path = f"{data_dir}/{name}_apriori.csv"
    metadata_path = f"metadata/preprocessing_metadata_{name}.json"
    rules_path = f"{rules_dir}/apriori_{name}.json"

    # load raw data
    df = pd.read_csv(raw_path, na_values=["?", "NaN"], sep=",")
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # clean headers and drop ID columns
    df = clean_header(df)
    df, dropped = drop_id_columns(df, extra_id_columns=cfg.get("extra_id_columns"))
    if dropped:
        print(f"Removed ID columns: {dropped}")

    df, split_cols = split_composite_columns(df, extra_split_columns=cfg.get("extra_split_columns"))
    if split_cols:
        print(f"Split composite columns: {split_cols}")

    target = _normalize_column_name(cfg["target"])

    df.to_csv(pre_path, index=False)

    # categorize/discretize the features
    df_cat, metadata = categorize(df, target=target)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # one-hot encoding
    df_enc = one_hot_encode(df_cat)
    df_enc.to_csv(apriori_path, index=False)

    # mine global association rules
    rule_set = mine_rules(df_enc, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE)
    rule_set.save(path=rules_path)
    print(f"Global rules saved to: {rules_path} ({len(rule_set)} rules)")

    # mine rules per target class
    target_columns = [c for c in df_enc.columns if target in c]
    print(f"Target columns found: {target_columns}")

    rule_sets = mine_rules_per_class(
        df_enc, target_columns, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE
    )
    for target_col, rs in rule_sets.items():
        path = f"{rules_dir}/apriori_{name}-{target_col}.json"
        rs.save(path=path)
        print(f"Rules for class '{target_col}' saved to: {path} ({len(rs)} rules)")


if __name__ == "__main__":
    for dataset_name, dataset_cfg in DATASETS.items():
        process_dataset(dataset_name, dataset_cfg)

    print("\nAll datasets were processed successfully.")