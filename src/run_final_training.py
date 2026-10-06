"""
Helper to rerun an experiment's final training without redoing the grid
search, reusing the best parameters found by a previous run.
"""

import json
import os
from pathlib import Path


DROP_KEYS = {"_score", "_metric", "kernel_initializer"}

DEFAULT_RESULTS_ROOTS = ["results"]


def _candidate_roots():
    env = os.environ.get("NARSY_PREV_RESULTS")
    roots = [env] if env else []
    roots += DEFAULT_RESULTS_ROOTS
    return [Path(r) for r in roots if r and Path(r).exists()]


def find_previous_config(dataset, variant, verbose=True):
    """Finds the experiment_config.json of the most recent run for (dataset, variant)."""
    for root in _candidate_roots():
        base = root / dataset / variant
        if not base.exists():
            continue

        runs = sorted(
            [d for d in base.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True
        )

        for run in runs:
            configs = list(run.glob("config/experiment_config.json")) + list(
                run.glob("experiment_config.json")
            )
            for cfg in configs:
                try:
                    data = json.loads(cfg.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if data.get("best_params"):
                    if verbose:
                        print(f"[fix_run] parameters from {cfg}")
                    return cfg, data["best_params"]

    return None, None


def get_params(dataset, variant, df_apriori=None, logger=None, required=True, drop_keys=None):
    """Returns the previous run's best_params, ready for the model.

    :param df_apriori: if passed, recomputes num_embedding_columns from the
                        data (never copy it from the old run: it depends on
                        the dataset).
    :param required: if True and no parameters are found, raises an error
                      instead of returning None.
    :param drop_keys: keys to discard. Defaults to DROP_KEYS, which also
                       drops kernel_initializer (needed for CARModel, which
                       already fixes it in code). For models that accept it,
                       pass {"_score", "_metric"}.
    :return: dict of parameters, or None if reuse is disabled.
    """
    if os.environ.get("NARSY_REUSE_PARAMS") != "1":
        return None

    cfg_path, params = find_previous_config(dataset, variant)

    if params is None:
        msg = (
            f"[fix_run] no best_params for {dataset}/{variant}. "
            f"Searched in: {[str(r) for r in _candidate_roots()]}"
        )
        if required:
            raise FileNotFoundError(msg)
        print(msg)
        return None

    params = {
        k: v
        for k, v in params.items()
        if k not in (DROP_KEYS if drop_keys is None else drop_keys)
    }

    if df_apriori is not None:
        params["num_embedding_columns"] = len(df_apriori.columns)

    if logger is not None:
        logger.info(f"Reused parameters from: {cfg_path}")
        logger.info(f"Parameters: {params}")

    return params