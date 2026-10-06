import json
import time
import gc
import numpy as np
import pandas as pd
import multiprocessing as mp
import itertools
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path
from typing import Iterable

from src import utils
from src.trainer import SimpleTrainer, CrossValTrainer, mine_rules_for_all_folds


# TODO: implement parameter sampling
# TODO: rename `scaler` to `scaler_cls`
def grid_search(model_class, data: pd.DataFrame, target: str, training_epochs: int,
                monitor='val_auc', apriori_data: pd.DataFrame = None,
                min_support: float = 0.2, min_confidence: float = 0.8,
                fixed_rules: dict = None,
                folds: int = None, splits: tuple = None,
                scaler=None, processes=8, digits=4, save: str = None,
                fixed_args: dict = None, seed=utils.SEED, save_as_csv=False, **kwargs):
    """
    :param model_class: a subclass of CARModel.
    :param data: the full dataset as data-frame.
    :param target: str, the name of the target variable.
    :param training_epochs: number of training epochs.
    :param monitor: metric to monitor during the search.
    :param apriori_data: one-hot encoded version of `data` (same row index) used to mine and
                         match association rules. Rules are mined PER FOLD, on training rows
                         only, to avoid leaking evaluation data into the mined knowledge.
                         Replaces the old `rules` argument, which passed a single dictionary
                         mined once over the whole dataset. Mutually exclusive with
                         `fixed_rules`.
    :param min_support: minimum support threshold for rule mining.
    :param min_confidence: minimum confidence threshold for rule mining.
    :param fixed_rules: optional rules-per-index dictionary (as returned by
                        `rules.RuleSet.get_per_index(...)`) mined ONCE, outside of this
                        function, and applied identically to every fold and every
                        configuration tried during the search. Use this only when you
                        deliberately want the same, pre-mined rule set for the whole
                        search (e.g. for consistency with a separate leakage-accepting
                        pipeline) -- NOT mined per-fold, so the same leakage caveat that
                        applies to a globally-mined rule set applies here too. Mutually
                        exclusive with `apriori_data`.
    :param folds: number of folds; Specify one between "folds" and "splits". If "folds" is set, the cross-validation
                  training will occur.
    :param splits: training-validation data-frames. Specify one between "folds" and "splits". If "splits" is set, a
                   simple train-validation training occurs.
    :param scaler: scaler class or callable.
    :param processes: number of parallel processes for multiprocessing.
    :param digits: number of rounding digits for logging.
    :param save: save path, optional. If set, all the tried parameter combinations will be saved in a file along their
                 validation score.
    :param fixed_args: fixed arguments not subject to the grid-search.
    :param seed: random seed.
    :param save_as_csv: whether to save the tried parameters and scores as a data-frame, instead of json.
    :param kwargs: hyperparameters on which grid-search is performed.
    """
    assert apriori_data is None or fixed_rules is None, \
        "Specify at most one of `apriori_data` (per-fold mining) and `fixed_rules` " \
        "(single pre-mined rule set): they are mutually exclusive."

    if isinstance(folds, int):
        folds_or_splits = int(folds)
    else:
        assert isinstance(splits, (tuple, list))
        folds_or_splits = splits

    if fixed_args is None:
        fixed_args = {}
    else:
        assert isinstance(fixed_args, dict), "Provide train-valid splits or k-folds!"

    precomputed_fold_rules = None

    if apriori_data is not None and isinstance(folds_or_splits, int):
        print(f'[grid_search] Mining association rules once for {folds_or_splits} folds '
              f'(reused across all configurations)...')

        precomputed_fold_rules = mine_rules_for_all_folds(
            data, folds=folds_or_splits, target_column=target,
            apriori_data=apriori_data, min_support=min_support,
            min_confidence=min_confidence, seed=seed)

    params_list = []

    for params in get_combinations(**kwargs):
        params.update(**fixed_args)
        params_list.append(params)

    args = [(model_class, training_epochs, data, folds_or_splits, params, target,
             scaler, precomputed_fold_rules, fixed_rules, monitor, seed) for params in params_list]

    results = []

    best_score = -np.inf
    best_params = None

    if int(processes) <= 1:

        bar = tqdm(args, total=len(args), desc='')

        for arg in bar:

            score, params = worker_job(arg)

            tf.keras.backend.clear_session()
            gc.collect()

            if score > best_score:
                best_score = score
                best_params = params

                bar.set_description(f'{monitor}: {round(best_score, digits)}')

            entry = {
                k: v for k, v in params.items()
            }

            entry['_score'] = score
            entry['_metric'] = monitor

            results.append(entry)

    else:

        with mp.Pool(processes=int(processes)) as pool:

            bar = tqdm(pool.imap(worker_job, args),total=len(args),desc='')

            for (score, params) in bar:

                if score > best_score:
                    best_score = score
                    best_params = params

                    bar.set_description(
                        f'{monitor}: {round(best_score, digits)}'
                    )

                entry = {
                    k: v for k, v in params.items()
                }

                entry['_score'] = score
                entry['_metric'] = monitor

                results.append(entry)

    if isinstance(save, str):
        path = Path(save)
        path.parent.mkdir(exist_ok=True)

        if save_as_csv:
            df = pd.DataFrame(results)
            df.to_csv(path, sep=',', index=False)
        else:
            json.dump(results, fp=open(path, 'w'))

    return best_score, best_params


def worker_job(args):

   
    (model_class, num_epochs, data, folds_or_splits, params, target, scaler,
     precomputed_fold_rules, fixed_rules, monitor, seed) = args

    start_time = time.time()

    utils.set_random_seed(seed, verbose=False)

    batch_size = params.pop('batch_size')


    if isinstance(folds_or_splits, int):
        # NOTE: `seed` is forwarded so the folds built here match the ones the rules were
        # mined on. Without it the trainer would fall back to its default seed and the
        # cached rules could belong to a different partition of the data.
        trainer = CrossValTrainer(data, folds=folds_or_splits, batch_sizes=(batch_size, 1024),
                                  target_column=target, scaler_cls=scaler,
                                  precomputed_fold_rules=precomputed_fold_rules,
                                  rules=fixed_rules,
                                  seed=seed)
    else:
        # SimpleTrainer still takes a single rules dictionary; when a fixed train/valid
        # split is used, mine the rules on the training split only and pass them here.
        trainer = SimpleTrainer(data, splits=folds_or_splits, batch_sizes=(batch_size, 1024),
                                target_column=target, scaler_cls=scaler, rules=fixed_rules)

    trainer.set_train_arguments(monitor=monitor, verbose=False)
    trainer.train(num_epochs, model_class, **params)

    if isinstance(trainer, CrossValTrainer):
        score = sum(trainer.scores) / len(trainer.scores)
    else:
        score = trainer.score

    params['batch_size'] = batch_size
    return score, params


def get_combinations(**kwargs) -> Iterable[dict]:
    combinations_per_param = [list(itertools.product([k], v)) for k, v in kwargs.items()]

    for combination in itertools.product(*combinations_per_param):
        combination_dict = {k: v for (k, v) in combination}
        yield combination_dict




















# import json
# import time
# import numpy as np
# import pandas as pd
# import multiprocessing as mp
# import itertools

# from tqdm import tqdm
# from pathlib import Path
# from typing import Iterable

# from src import utils
# from src.trainer import SimpleTrainer, CrossValTrainer


# # TODO: implement parameter sampling
# # TODO: rename `scaler` to `scaler_cls`
# def grid_search(model_class, data: pd.DataFrame, target: str, training_epochs: int,
#                 monitor='val_auc', rules=None, folds: int = None, splits: tuple = None,
#                 scaler=None, processes=8, digits=4, save: str = None,
#                 fixed_args: dict = None, seed=utils.SEED, save_as_csv=False, **kwargs):
#     """
#     :param model_class: a subclass of CARModel.
#     :param data: the full dataset as data-frame.
#     :param target: str, the name of the target variable.
#     :param training_epochs: number of training epochs.
#     :param monitor: metric to monitor during the search.
#     :param rules: rules by index dictionary.
#     :param folds: number of folds; Specify one between "folds" and "splits". If "folds" is set, the cross-validation
#                   training will occur.
#     :param splits: training-validation data-frames. Specify one between "folds" and "splits". If "splits" is set, a
#                    simple train-validation training occurs.
#     :param scaler: scaler class or callable.
#     :param processes: number of parallel processes for multiprocessing.
#     :param digits: number of rounding digits for logging.
#     :param save: save path, optional. If set, all the tried parameter combinations will be saved in a file along their
#                  validation score.
#     :param fixed_args: fixed arguments not subject to the grid-search.
#     :param seed: random seed.
#     :param save_as_csv: whether to save the tried parameters and scores as a data-frame, instead of json.
#     :param kwargs: hyperparameters on which grid-search is performed.
#     """
#     if isinstance(folds, int):
#         folds_or_splits = int(folds)
#     else:
#         assert isinstance(splits, (tuple, list))
#         folds_or_splits = splits

#     if fixed_args is None:
#         fixed_args = {}
#     else:
#         assert isinstance(fixed_args, dict), "Provide train-valid splits or k-folds!"

#     # prepare arguments for the parallel processing
#     params_list = []

#     for params in get_combinations(**kwargs):
#         params.update(**fixed_args)
#         params_list.append(params)

#     args = [(model_class, training_epochs, data, folds_or_splits, params, target,
#              scaler, rules, monitor, seed) for params in params_list]

#     results = []

#     best_score = -np.inf
#     best_params = None

#     # SERIAL EXECUTION (GPU SAFE)
#     if int(processes) <= 1:

#         bar = tqdm(args, total=len(args), desc='')

#         for arg in bar:

#             score, params = worker_job(arg)

#             if score > best_score:
#                 best_score = score
#                 best_params = params

#                 bar.set_description(
#                     f'{monitor}: {round(best_score, digits)}'
#                 )

#             entry = {
#                 k: v for k, v in params.items()
#             }

#             entry['_score'] = score
#             entry['_metric'] = monitor

#             results.append(entry)

#     # MULTIPROCESS EXECUTION
#     else:

#         with mp.Pool(processes=int(processes)) as pool:

#             bar = tqdm(
#                 pool.imap(worker_job, args),
#                 total=len(args),
#                 desc=''
#             )

#             for (score, params) in bar:

#                 if score > best_score:
#                     best_score = score
#                     best_params = params

#                     bar.set_description(
#                         f'{monitor}: {round(best_score, digits)}'
#                     )

#                 entry = {
#                     k: v for k, v in params.items()
#                 }

#                 entry['_score'] = score
#                 entry['_metric'] = monitor

#                 results.append(entry)

#     if isinstance(save, str):
#         path = Path(save)
#         path.parent.mkdir(exist_ok=True)

#         if save_as_csv:
#             df = pd.DataFrame(results)
#             df.to_csv(path, sep=',', index=False)
#         else:
#             json.dump(results, fp=open(path, 'w'))

#     return best_score, best_params


# def worker_job(args):

   
#     (model_class, num_epochs, data, folds_or_splits, params, target, scaler,
#      rules, monitor, seed) = args

#     start_time = time.time()

#     utils.set_random_seed(seed, verbose=False)

#     batch_size = params.pop('batch_size')


#     if isinstance(folds_or_splits, int):
#         trainer = CrossValTrainer(data, folds=folds_or_splits, batch_sizes=(batch_size, 1024),
#                                   target_column=target, scaler_cls=scaler, rules=rules)
#     else:
#         trainer = SimpleTrainer(data, splits=folds_or_splits, batch_sizes=(batch_size, 1024),
#                                 target_column=target, scaler_cls=scaler, rules=rules)

#     trainer.set_train_arguments(monitor=monitor, verbose=False)
#     trainer.train(num_epochs, model_class, **params)


 

#     elapsed = (time.time() - start_time) / 60

#     print(
#         f"CONFIGURATION COMPLETED IN {elapsed:.2f} MINUTES"
#     )


#     if isinstance(trainer, CrossValTrainer):
#         score = sum(trainer.scores) / len(trainer.scores)
#     else:
#         score = trainer.score

#     params['batch_size'] = batch_size
#     return score, params


# def get_combinations(**kwargs) -> Iterable[dict]:
#     combinations_per_param = [list(itertools.product([k], v))
#                               for k, v in kwargs.items()]

#     for combination in itertools.product(*combinations_per_param):
#         combination_dict = {k: v for (k, v) in combination}
#         yield combination_dict
