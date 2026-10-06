"""
src/preprocessing.py

Data cleaning, removal of ID columns, and discretization of numerical features.
No external discretization library: equal-frequency binning is
implemented directly using pandas.qcut.

Public functions, in the order they are typically used in a pipeline:
    clean_header(df)                -> normalizes column names
    drop_id_columns(df, extra=None) -> removes ID columns
    categorize(df, target=None)     -> converts all columns to categorical/string
    one_hot_encode(df)              -> Boolean one-hot encoding
"""


import pandas as pd


DEFAULT_ID_COLUMNS = {
    "id", "user_id", "customer_id", "patient_id", "identifier", "index",
    "customerid", "person_id", "assessmentid", "case_id",
    "patient_number", "number",
}

SEMANTIC_LABELS = {
    2: ["low", "high"],
    3: ["low", "medium", "high"],
    4: ["very_low", "low", "high", "very_high"],
    5: ["very_low", "low", "medium", "high", "very_high"],
}

MAX_BINS = 5          
UNIQUE_THRESHOLD = 7   


def clean_header(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizza i nomi delle colonne: spazi/trattini -> underscore, rimuove simboli."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def drop_id_columns(df: pd.DataFrame, extra_id_columns=None):
    """
    Removes the identifying columns (case-insensitive).

    extra_id_columns: optional list of dataset-specific extra names
    (e.g., ID columns with names not included in DEFAULT_ID_COLUMNS).

    Returns (df_without_ids, list_of_removed_columns).
    """
    id_names = set(DEFAULT_ID_COLUMNS)
    if extra_id_columns:
        id_names |= {c.strip().lower() for c in extra_id_columns}

    to_drop = [c for c in df.columns if c.strip().lower() in id_names]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df, to_drop


def _choose_num_bins(series: pd.Series) -> int:
    n_unique = series.nunique()
    return max(2, min(MAX_BINS, n_unique))


def _discretize_series(series: pd.Series) -> pd.Series:
    """Equal-frequency binning with semantic labels (low/medium/high...)."""
    k = _choose_num_bins(series)
    while k >= 2:
        labels = SEMANTIC_LABELS.get(k, [f"class_{i}" for i in range(k)])
        try:
            return pd.qcut(series, q=k, labels=labels, duplicates="drop").astype(str)
        except ValueError:
            k -= 1
    return series.astype(str)



SPLIT_COLUMNS = {
    "blood_pressure": ("/", ["Systolic", "Diastolic"]),
}


def split_composite_columns(df: pd.DataFrame, extra_split_columns: dict = None):
    """
    Splits the "composite" columns (e.g., Blood_Pressure "120/80" -> Systolic, Diastolic)
    into separate numeric columns, and removes the original column.

    extra_split_columns: optional dict {column_name: (separator, [name1, name2])}
    to add dataset-specific composite columns without touching the default.
    Returns (df_transformed, list_of_splitted_columns).
    """
    df = df.copy()
    split_map = dict(SPLIT_COLUMNS)
    if extra_split_columns:
        split_map.update(extra_split_columns)

    splitted = []
    for col in list(df.columns):
        key = col.strip().lower()
        if key not in split_map:
            continue

        sep, new_names = split_map[key]
        parts = df[col].astype(str).str.split(sep, expand=True)

        if parts.shape[1] != len(new_names):
            continue

        for i, new_name in enumerate(new_names):
            df[new_name] = pd.to_numeric(parts[i], errors="coerce")

        df = df.drop(columns=[col])
        splitted.append(col)

    return df, splitted


def categorize(df: pd.DataFrame, target: str = None, unique_threshold: int = UNIQUE_THRESHOLD):
    """
    Transforms all columns into categorical features (strings), ready for
    one-hot encoding.

    - the target column (if specified) is NOT discretized, but is
      nevertheless cast to a string: this allows it to be properly expanded into
      class-specific columns by the one-hot encoding step (necessary for multi-class targets,
      e.g., values 1/2/3 as in NPHA)
    - numeric columns with few unique values (<= unique_threshold) remain unchanged
      (they are already likely ordinal/binary, e.g., Likert scale 1-5)
    - continuous numeric columns are discretized into equal-frequency bins
    - already textual/categorical columns remain unchanged (cast to string)

    Returns (df_transformed, metadata) where metadata is a dict {column: info}.
    """
    df = df.copy()
    metadata = {}

    for col in df.columns:
        if target is not None and col == target:
            df[col] = df[col].astype(str)
            metadata[col] = {"role": "target"}
            continue

        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique()
            if n_unique <= unique_threshold:
                df[col] = series.astype(str)
                metadata[col] = {
                    "type": "numeric_low_cardinality",
                    "action": "kept",
                    "n_unique": int(n_unique),
                }
            else:
                df[col] = _discretize_series(series)
                metadata[col] = {"type": "numeric_continuous", "action": "discretized"}
        else:
            df[col] = series.astype(str)
            metadata[col] = {"type": "categorical", "action": "kept"}

    return df, metadata


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df).astype(bool)







