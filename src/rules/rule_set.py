import json
import numpy as np
import pandas as pd

from src.rules import rule_utils

from tqdm import tqdm
from typing import List, Dict, Optional, Union
from pathlib import Path


class RuleSet:
    """Helper object that wraps a pd.DataFrame of rules"""

    def __init__(self, rules: Union[list, pd.DataFrame]):
        self.rules = pd.DataFrame(rules)
        self.per_index: Dict[int, list] = None
        self.frontiers: List[pd.DataFrame] = []

    @classmethod
    def load(cls, path: str) -> 'RuleSet':
        assert Path(path).exists(), f'Given path "{path}" does not exists!'

        rules = json.load(fp=open(path))
        return cls(rules)

    def __len__(self):
        return len(self.rules)

    def save(self, path: str):
        rules_dict = {k: list(self.rules[k]) for k in self.rules.keys()}
        rules_dict['antecedents'] = [list(v) for v in rules_dict['antecedents']]
        rules_dict['consequents'] = [list(v) for v in rules_dict['consequents']]

        json.dump(rules_dict, fp=open(path, 'w'))

    def get_per_index(self, *args, **kwargs):
        raise NotImplementedError

    def filter_by_rhs(self, attribute: str, inplace=True) -> 'RuleSet':
        mask = self.rules['consequents'].apply(lambda x: attribute in x)

        if inplace:
            self.rules = self.rules[mask]
            return self

        return self.__class__(rules=self.rules[mask].copy())

    def filter_by_lhs(self, attributes: List[str], inplace=True) -> 'RuleSet':
        mask = self.rules['antecedents'].apply(lambda x: any(a in x for a in attributes))

        if inplace:
            self.rules = self.rules[mask]
            return self

        return self.__class__(rules=self.rules[mask].copy())

    def filter_by_metric(self, metric: str, min_value: float = None, max_value: float = None, inplace=False) -> 'RuleSet':
        assert metric in self.rules.columns, f'Metric "{metric}" not available!'

        if isinstance(min_value, float):
            mask = self.rules[metric] > min_value

            if isinstance(max_value, float):
                mask &= self.rules[metric] < max_value

        elif isinstance(max_value, float):
            mask = self.rules[metric] < max_value
        else:
            raise ValueError('Provide at least one between `min_value` and `max_value`!')

        if inplace:
            self.rules = self.rules[mask]
            return self

        return self.__class__(self.rules[mask].copy())

    def take_top_k(self, metric: str, top_k: Union[int, float], inplace=False, **sort_kwargs) -> 'RuleSet':
        assert metric in self.rules.columns, f'Metric "{metric}" not available!'

        if isinstance(top_k, int):
            amount = min(top_k, len(self))
        else:
            assert isinstance(top_k, float)
            assert 0.0 < top_k < 1.0
            amount = int(len(self) * top_k)

        sorted_rules = self.rules.sort_values(by=metric, inplace=False, **sort_kwargs)
        index = sorted_rules.index[:amount]

        if inplace:
            self.rules = self.rules.loc[index]
            return self

        return self.__class__(self.rules.loc[index].copy())

    def filter_rhs(self, keep_values: list):
        """Discards all the RHS' values that are not in `keep_values`."""
        assert len(keep_values) >= 1

        new_rhs = self.rules['consequents'].apply(lambda rhs: [x for x in rhs if x in keep_values])
        mask = new_rhs.apply(lambda x: len(x)) != 0

        self.rules['consequents'] = new_rhs
        self.rules = self.rules[mask]

    def take(self, k: int, inplace=False) -> 'RuleSet':
        """Takes the first `k` rules"""
        k = int(k)
        assert k >= 0

        if inplace:
            self.rules = self.rules[:k]
            return self

        return self.__class__(rules=self.rules[:k].copy())

    def compute_pareto_frontiers(self, criteria: List[str]):
        pareto_ranks = rule_utils.pareto_frontiers(self.rules, criteria,
                                                   inplace=False)
        frontiers = rule_utils.split_rules_by_frontier(self.rules, pareto_ranks)

        for f in frontiers:
            self.frontiers.append(f)

    def classify(self, df: pd.DataFrame, target: Optional[str] = None, save_scores=False) -> np.ndarray:
        accuracies = np.zeros(shape=len(self.rules), dtype=np.float32)

        for i, (_, rule) in enumerate(self.rules.iterrows()):
            if target is None: 
                rhs = list(rule['consequents'])[0] 
                acc = self._accuracy_score(df, target=rhs, rule_lhs=rule['antecedents'])
            else:
                acc = self._accuracy_score(df, target=target, rule_lhs=rule['antecedents'])
            accuracies[i] = acc

        if save_scores:
            self.rules['accuracy'] = accuracies

        return accuracies

  
    def regress(self, df: pd.DataFrame, df_apriori: pd.DataFrame, target: Optional[str] = None, save_scores=False):
        scores = np.zeros(shape=len(self.rules), dtype=np.float32)

        for i, (_, rule) in tqdm(enumerate(self.rules.iterrows()), total=len(self)):
            if target is None:
                rhs = list(rule['consequents'])[0]
                mse = self._mse_score(df, df_apriori, target=rhs, rule_lhs=rule['antecedents'])
            else:
                mse = self._mse_score(df, df_apriori, target=target, rule_lhs=rule['antecedents'])
            scores[i] = mse

        if save_scores:
            self.rules['mean_squared_error'] = scores

        return scores



    @staticmethod
    def _accuracy_score(df: pd.DataFrame, target: str, rule_lhs: List[str]) -> float:
        predictions = df[rule_lhs[0]].copy()

        for lhs in rule_lhs[1:]:
            predictions *= df[lhs].copy()

        accuracy = np.array(predictions == df[target], dtype=np.float32)
        return float(accuracy.mean())

    @staticmethod
    def _mse_score(df: pd.DataFrame, df_apriori: pd.DataFrame, target: str,
                   rule_lhs: List[str]) -> float:
        mask = df_apriori[rule_lhs[0]] == 1

        for lhs in rule_lhs[1:]:
            mask &= df_apriori[lhs] == 1

        if mask.sum() <= 0:
            return np.inf

        indices = df_apriori[mask].index
        targets = df.loc[indices][target]
        predictions = targets.mean()

        mse = np.square(targets - predictions).mean()
        return float(mse)

    def __add__(self, other: 'RuleSet') -> 'RuleSet':
        return self.union(other)

    def union(self, other: 'RuleSet'):
        assert isinstance(other, RuleSet) or issubclass(other.__class__, RuleSet)

        rules_union = pd.concat([self.rules, other.rules])
        return self.__class__(rules=rules_union)


class AprioriRuleSet(RuleSet):

    @classmethod
    def load(cls, path: str) -> 'RuleSet':
        return super().load(path)
    

    def get_per_index(self, df: pd.DataFrame, top_k=None, return_coverage=False) -> Union[Dict[int, list], tuple]:

        if isinstance(self.per_index, dict):

            if return_coverage:
                coverage = len(self.per_index) / len(df) * 100
                return self.per_index, coverage

            return self.per_index

        rules_per_index = rule_utils.get_apriori_rules_per_index(
             df, rules=self.rules, top_k=top_k
        )

        coverage = len(rules_per_index) / len(df) * 100
        print(f'Indices cover {round(coverage, 2)}% of data.')

        self.per_index = rules_per_index

        if return_coverage:
            return rules_per_index, coverage

        return rules_per_index


    def get_per_index_cov(self, df: pd.DataFrame, top_k=None, return_coverage=False) -> Union[Dict[int, list], tuple]:

        if isinstance(self.per_index, dict):

            if return_coverage:
                coverage = len(self.per_index) / len(df) * 100
                return self.per_index, coverage

            return self.per_index

        rules_per_index = rule_utils.count_rules_per_row(df, rules=self.rules, top_k=top_k)

        coverage = len(rules_per_index) / len(df) * 100
        print(f'Indices cover {round(coverage, 2)}% of data.')

        self.per_index = rules_per_index

        if return_coverage:
            return rules_per_index, coverage

        return rules_per_index
