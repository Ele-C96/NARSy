import gc
import time
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import openml

from typing import Union, List, Tuple
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold


SEED = 42
LOGGER = logging.getLogger('arl-ml')
LOGGER.setLevel(logging.WARNING)


def set_random_seed(seed: int, verbose=True):
    global SEED

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

        SEED = seed

        if verbose:
            print(f'Random seed {SEED} set.')


def get_random_generator(seed=SEED, generator=None) -> np.random.Generator:
    if seed is not None:
        seed = int(seed)
        assert 0 <= seed < 2 ** 32

    if generator is None or not isinstance(generator, np.random.BitGenerator):
        generator = np.random.MT19937

    return np.random.default_rng(generator(seed=seed))


def free_mem():
    return gc.collect()


def replace_whitespace(string: str, symbol='_'):
    return string.strip().replace(' ', str(symbol))


def create_directories(path: str):
    
    path = Path(path)

    for parent in list(path.parents)[::-1][1:]:
        assert parent.is_dir()

        if not parent.exists():
            parent.mkdir()

    if path.is_dir():
        path.mkdir(exist_ok=True)


def get_header_map(headers: list, preprocess=True, **kwargs) -> dict:
    return {f'A{i}': replace_whitespace(h, **kwargs) if preprocess else h for i, h in enumerate(headers)}


def map_node(node, header_map: dict):
    return [header_map[n] for n in node]


def get_num_combinations(n: int, from_k=2, to_k=None):
    to_k = n if to_k is None else to_k + 1
    num = 0

    for k in range(from_k, to_k):
        num += math.comb(n, k)

    return num


def get_openml_dataset(name: str) -> Tuple[pd.DataFrame, List[str]]:
    ds = openml.datasets.get_dataset(name)
    df, _, _, header = ds.get_data()
    return df, header


class TimeIt:
   
    t0: float
    t1: float
    t_sec: int

    def __enter__(self):
        self.t0 = time.time_ns()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = time.time_ns() - self.t0
        self.t_sec = self._convert_time_in_seconds(self.t1)
        print(f'Elapsed {self.t_sec}s')

    @staticmethod
    def _convert_time_in_seconds(t: float, digits=2):
        return round(t / (10**9), ndigits=digits)


def to_transactions(df: pd.DataFrame, remove_nans=False) -> List[tuple]:
    if remove_nans:
        df = df[~df.isna().any()]

    return [tuple(row) for row in df.values.tolist()]


def tf_global_norm(tensors: list, **kwargs):
    norms = [tf.norm(x, **kwargs) for x in tensors]
    return tf.sqrt(tf.reduce_sum([norm * norm for norm in norms]))


def k_fold_splits(df: pd.DataFrame, folds: int, stratify=False, target_column: str = None,
                  shuffle=True, return_indices=False,
                  seed=SEED) -> Union[tuple, List[Tuple[pd.DataFrame, pd.DataFrame]]]:
    folds = int(folds)
    assert folds > 1, 'Provide at the least two folds for cross-validation splitting!'

    if stratify:
        assert isinstance(target_column, str), 'Must provide target column when `stratify=True`!'

        splitter = StratifiedKFold(n_splits=folds, shuffle=bool(shuffle), random_state=seed)
        indices = splitter.split(df, df[target_column])
    else:
        splitter = KFold(n_splits=folds, shuffle=bool(shuffle), random_state=seed)
        indices = splitter.split(df)

    splits = []

    for (train_indices, test_indices) in indices:
        splits.append((df.iloc[train_indices], df.iloc[test_indices]))

    if return_indices:
        return splits, indices

    return splits
