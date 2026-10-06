import os
import glob
import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path

from mlxtend.frequent_patterns import apriori, association_rules

from src import utils
from src.models import DataSequence
from src.models.sequences import AprioriSequence
from src.rules import AprioriRuleSet


class Trainer:
    """Abstract base class to define a model trainer."""

    def __init__(self, data: pd.DataFrame, seed=utils.SEED):
        self.seed = seed
        self.data = data
        self.has_categorical_features = any(data.dtypes == 'object')
        self.train_args = dict()

    def set_train_arguments(self, **kwargs):
        """Function to set additional arguments passed to `CARModel.train()`"""
        self.train_args.update(**kwargs)

    def train(self, *args, **kwargs):
        raise NotImplemented

    def get_categorical_map(self) -> dict:
        raise NotImplemented

    def load(self, *args, **kwargs):
        raise NotImplemented


class SimpleTrainer(Trainer):
    """A simple train-validation trainer"""

    def __init__(self, data: pd.DataFrame, splits: tuple, batch_sizes: tuple, target_column: str, scaler_cls=None, rules=None, seed=utils.SEED):
        """
        :param data: the full, not split, dataset as a pd.DataFrame.
        :param splits: a size-two tuple of data-frames. The first element is the train-set, the second the
                       validation-set.
        :param batch_sizes: a size-two tuple of batch sizes, one for training and the other for validation.
        :param target_column: str, the name of the target column.
        :param scaler_cls: the class of a sklearn.Scaler or a custom, argument-free, callable.
        :param rules: the rules per index dictionary, obtained by `rules.RuleSet.get_rules_per_index(...)`.
        :param seed: the random seed, defaults to the global one if set.
        """
        assert len(splits) == 2, "train-valid splits needed!"
        super().__init__(data, seed)

        self.score = 0.0
        self.model = None

        self.train_seq = DataSequence(data=splits[0], rules=rules, batch_size=batch_sizes[0], scaler=None if scaler_cls is None else scaler_cls(), target_column=str(target_column), shuffle=True, seed=self.seed)

        self.valid_seq = DataSequence(data=splits[1], rules=rules, batch_size=batch_sizes[1], seed=self.seed, target_column=self.train_seq.target_column, scaler=self.train_seq.scaler, num_classes=self.train_seq.num_classes, deterministic_rules=True)

    def train(self, num_epochs: int, model_class, digits=4, **model_kwargs):
        """
        :param num_epochs: number of training epochs.
        :param model_class: the class of the model, must be a subclass of `CARModel`.
        :param digits: number of rounding digits when printing the validation score.
        :param model_kwargs: arguments passed to create and compile the model, see `CARModel.get_compiled()`.
        """
        utils.set_random_seed(self.seed)
        categorical_map = self.get_categorical_map()

        self.model = model_class.get_compiled(self.data,rule_max_len=self.train_seq.rule_max_len,target_column=self.train_seq.target_column,categorical_columns=categorical_map,**model_kwargs)

        self.score = self.model.train(self.train_seq, epochs=num_epochs, **self.train_args, simple=True, validation_sequence=self.valid_seq)

        if self.train_args.get('verbose', False):
            print(f'Training score: {round(self.score, digits)}')

    def save(self, name: str, base_dir=None, digits=4):
        """Save model's weights and history"""
        utils.create_directories(base_dir)

        pattern = f'{name}-{round(self.score, digits)}'
        self.model.save_model(f'weights-{pattern}.keras', f'history-{pattern}.json', base_dir)

    def get_categorical_map(self) -> Optional[dict]:
        if self.has_categorical_features:
            return self.train_seq.get_categorical_columns_mapping()

        return None

    def load(self, weights_path: str, model_class, **model_kwargs):
        assert Path(weights_path).exists(), f'Weights at "{weights_path}" do not exist!'
        categorical_map = self.get_categorical_map()

        self.model = model_class.get_compiled(self.data, rule_max_len=self.train_seq.rule_max_len, target_column=self.train_seq.target_column, categorical_columns=categorical_map, **model_kwargs)
        self.model.load_weights(weights_path)
        print(f'Loaded weights at "{weights_path}".')



def mine_fold_rules(train_df: pd.DataFrame, test_df: pd.DataFrame,apriori_data: pd.DataFrame, min_support: float = 0.2, min_confidence: float = 0.8, verbose=True):
    """Mines association rules using ONLY this fold's training rows, then matches those
    rules (without re-mining) against both the training and the test rows of this same
    fold.

    Returns (rule_set, rules_per_index).
    """
    train_apriori = apriori_data.loc[train_df.index]

    frequent_itemsets = apriori(train_apriori, min_support=min_support,use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if verbose:
        print(f'[mine_fold_rules] {len(train_apriori)} training rows -> 'f'{len(frequent_itemsets)} frequent itemsets -> {len(rules)} rules.')

    rule_set = AprioriRuleSet(rules)

    fold_index = train_df.index.union(test_df.index)
    fold_apriori = apriori_data.loc[fold_index]

    rules_per_index = rule_set.get_per_index(fold_apriori)
    return rule_set, rules_per_index


def mine_rules_for_all_folds(data: pd.DataFrame, folds: int, target_column: str,apriori_data: pd.DataFrame, min_support: float = 0.2,min_confidence: float = 0.8, stratify=False, shuffle_folds=True, seed=utils.SEED) -> list:
    """Mines the per-fold rules once, so they can be reused across many model
    configurations (e.g. during a grid search) instead of being re-mined for each one.
    """
    splits = utils.k_fold_splits(data, folds, stratify=stratify, target_column=str(target_column), shuffle=shuffle_folds, seed=None if not shuffle_folds else seed)

    return [mine_fold_rules(train_df, test_df, apriori_data, min_support, min_confidence) for train_df, test_df in splits]


class CrossValTrainer(Trainer):
    

    def __init__(self, data: pd.DataFrame, folds: int, batch_sizes: tuple,target_column: str, apriori_data: pd.DataFrame = None,min_support: float = 0.2, min_confidence: float = 0.8,precomputed_fold_rules: list = None,rules: dict = None,stratify=False, scaler_cls=None, shuffle_folds=True, seed=utils.SEED):
        """
        :param apriori_data: one-hot encoded version of `data` (same row index), used to mine
                             and match association rules. If None, no rules are used at all.
        :param min_support: minimum support threshold passed to `mlxtend.apriori`.
        :param min_confidence: minimum confidence threshold passed to `mlxtend.association_rules`.
        :param precomputed_fold_rules: optional list of (rule_set, rules_per_index), one per fold, as returned by `mine_rules_for_all_folds`. Use it
                                       to avoid re-mining identical rules for every model configuration during a grid search. The split arguments must match those used when mining.
        """
        super().__init__(data, seed)

        self.apriori_data = apriori_data
        self.min_support = min_support
        self.min_confidence = min_confidence

        self.splits = utils.k_fold_splits(data, folds, stratify=stratify,target_column=str(target_column),shuffle=shuffle_folds, seed=None if not shuffle_folds else self.seed)

        if precomputed_fold_rules is not None:
            assert len(precomputed_fold_rules) == len(self.splits), \
                (f'Got {len(precomputed_fold_rules)} precomputed rule sets but '
                 f'{len(self.splits)} folds: the cached rules were mined on a different '
                 f'split configuration.')

        self.fold_rule_sets = []
        self.sequences = []

        for i, (train_df, test_df) in enumerate(self.splits):
            rules_per_index = None

            if precomputed_fold_rules is not None:
                rule_set, rules_per_index = precomputed_fold_rules[i]
                self.fold_rule_sets.append(rule_set)

            elif self.apriori_data is not None:
                rule_set, rules_per_index = mine_fold_rules(
                    train_df, test_df, self.apriori_data,
                    self.min_support, self.min_confidence)
                self.fold_rule_sets.append(rule_set)

            elif rules is not None:             
                rules_per_index = rules

            train_seq = DataSequence(train_df, rules=rules_per_index, batch_size=batch_sizes[0],target_column=str(target_column), shuffle=True, seed=self.seed, scaler=None if scaler_cls is None else scaler_cls())

            if train_seq.num_targets > 0:
                num_targets = train_seq.num_targets
                num_classes = None
            else:
                assert train_seq.num_classes > 0
                num_targets = None
                num_classes = train_seq.num_classes

            test_seq = DataSequence(test_df, rules=rules_per_index,batch_size=batch_sizes[1],target_column=str(target_column),shuffle=False, seed=self.seed,scaler=train_seq.scaler,num_classes=num_classes,num_targets=num_targets,deterministic_rules=True)

            self.sequences.append((train_seq, test_seq))

        self.scores = []
        self.models = []

    def train(self, num_epochs: int, model_class, **model_kwargs):
        categorical_map = self.get_categorical_map()

        for i, (train_seq, valid_seq) in enumerate(self.sequences):
           

            utils.set_random_seed(self.seed)

            model = model_class.get_compiled(self.splits[i][0], rule_max_len=train_seq.rule_max_len,target_column=train_seq.target_column,categorical_columns=categorical_map,**model_kwargs)

            score = model.train(train_seq, epochs=num_epochs, **self.train_args,validation_sequence=valid_seq, simple=True)
            self.scores.append(score)
            self.models.append(model)

        if self.train_args.get('verbose', False):
            print(f'Score: {np.round(np.mean(self.scores, 2))} (avg)'
                  f' ± {np.round(np.std(self.scores))} (std)')

    def save(self, name: str, base_dir=None, remove_old=False, digits=3):
        utils.create_directories(base_dir)

        if remove_old:
            for file in glob.glob(f'{base_dir}/weights_*-{name}-*.keras'):
                print(f'removing "{file}"')
                os.remove(file)

            for file in glob.glob(f'{base_dir}/history_*-{name}-*.json'):
                print(f'removing "{file}"')
                os.remove(file)

        for fold, (model, score) in enumerate(zip(self.models, self.scores)):
            pattern = f'{fold}-{name}-{round(score, digits)}'

            model.save_model(f'weights_{pattern}.keras', f'history_{pattern}.json', base_dir)

    def get_categorical_map(self) -> Optional[dict]:
        if not self.has_categorical_features:
            return None

        categorical_map = self.sequences[0][0].get_categorical_columns_mapping()

        for seq, _ in self.sequences[1:]:
            other_map = seq.get_categorical_columns_mapping()

            for k, v in categorical_map.items():
                categorical_map[k]['num_categories'] = max(v['num_categories'], other_map[k]['num_categories'])
        return categorical_map



class AprioriCrossValTrainer(CrossValTrainer):
   

    def __init__(self, data: pd.DataFrame, apriori_data: pd.DataFrame,folds: int, batch_sizes: tuple,target_column: str, min_support: float = 0.2, min_confidence: float = 0.8,precomputed_fold_rules: list = None,stratify=False, scaler_cls=None, shuffle_folds=True, seed=utils.SEED):
        super().__init__(data, folds, batch_sizes, target_column, apriori_data=apriori_data, min_support=min_support, min_confidence=min_confidence, precomputed_fold_rules=precomputed_fold_rules, stratify=stratify,scaler_cls=scaler_cls, shuffle_folds=shuffle_folds, seed=seed)

        self.splits = [(train, test,apriori_data.loc[train.index],apriori_data.loc[test.index]) for train, test in self.splits]

        rebuilt_sequences = []

        for (train_df, test_df, apriori_train, apriori_test), (train_seq, test_seq) \
                in zip(self.splits, self.sequences):

            new_train_seq = AprioriSequence(train_df, apriori_train, rules=train_seq.rules,batch_size=batch_sizes[0], target_column=str(target_column),shuffle=True, seed=self.seed, scaler=None if scaler_cls is None else scaler_cls())

            if new_train_seq.num_targets > 0:
                num_targets, num_classes = new_train_seq.num_targets, None
            else:
                num_targets, num_classes = None, new_train_seq.num_classes

            new_test_seq = AprioriSequence(test_df, apriori_test, rules=test_seq.rules, batch_size=batch_sizes[1], target_column=str(target_column),shuffle=False, seed=self.seed,scaler=new_train_seq.scaler,num_classes=num_classes, num_targets=num_targets,deterministic_rules=True)

            rebuilt_sequences.append((new_train_seq, new_test_seq))

        self.sequences = rebuilt_sequences

