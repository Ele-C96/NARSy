"""
Adaptive Minimum Support Threshold
====================================
Implementation based on:
Hikmawati, E., Maulidevi, N.U., Surendro, K. (2021).
"Minimum threshold determination method based on dataset characteristics
in association rule mining". Journal of Big Data, 8:146.
https://doi.org/10.1186/s40537-021-00538-3

IMPORTANT METHODOLOGICAL NOTE
-----------------------------
The paper presents two formulations; this implementation uses
VERSION 2, since it is the one empirically verified against the
published results.
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None


def adaptive_min_support(
    transactions: List[Iterable[str]],
    item_utility: Optional[Dict[str, float]] = None,
    return_density: bool = False,
):
    """
    Computes the minimum support threshold adaptively, following
    Hikmawati et al. (2021), using the formula verified against the
    published experimental results (Table 3 of the paper).

    Parameters
    ----------
    transactions : list of transactions, each an iterable of items (str)
    item_utility : optional dict {item: weight}. If None, every item has
                   utility 1 (pure "adaptive support" case, based only on
                   the dataset's density -- this is the case tested in
                   the paper, where the utility criterion is fixed to 1
                   for every item).
    return_density : if True, also returns the density value
                      (useful for comparing against Table 3 of the paper).

    Returns
    -------
    min_sup : float, proposed minimum support threshold (in [0, 1])
    density : float, optional, returned only if return_density=True
    """
    n_transactions = len(transactions)
    if n_transactions == 0:
        raise ValueError("The transaction dataset is empty.")

    freq = Counter()
    for t in transactions:
        freq.update(set(t))

    n_items = len(freq)
    if n_items == 0:
        raise ValueError("No items found in the transactions.")

    if item_utility is None:
        item_utility = {item: 1.0 for item in freq}

    weighted_freq_sum = sum(
        n_d * item_utility.get(item, 1.0) for item, n_d in freq.items()
    )
    density = weighted_freq_sum / n_items

    min_sup = density / n_transactions

    if return_density:
        return min_sup, density
    return min_sup


def dataframe_to_transactions(
    df,
    n_bins: int = 5,
    numeric_columns: Optional[List[str]] = None,
    binning: str = "quantile",
) -> List[List[str]]:
    """
    Converts a tabular DataFrame into a list of transactions (lists of items),
    automatically discretizing the numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
    n_bins : number of bins for the numeric columns
    numeric_columns : list of columns to treat as numeric;
                       if None, they are detected automatically
    binning : 'quantile' (equal-frequency) or 'uniform' (equal-width)

    Returns
    -------
    transactions : list of lists of items (strings "column_value")
    """
    if pd is None:
        raise ImportError("pandas is required for this function (pip install pandas).")

    df = df.copy()
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_columns:
        strategy = "quantile" if binning == "quantile" else "uniform"
        if strategy == "quantile":
            df[col] = pd.qcut(df[col], q=n_bins, duplicates="drop")
        else:
            df[col] = pd.cut(df[col], bins=n_bins)
        df[col] = df[col].astype(str)

    transactions = []
    for _, row in df.iterrows():
        items = [f"{col}={val}" for col, val in row.items()]
        transactions.append(items)

    return transactions