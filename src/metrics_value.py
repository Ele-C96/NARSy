from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from pandas import DataFrame
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from src.rules import AprioriRuleSet, rule_utils, rule_set


class RuleSetMetricAnalyzer:
    

    def __init__(self, ruleset: AprioriRuleSet, data: DataFrame, data_base: DataFrame):
        self.R = ruleset
        self.D = data
        self.D_base = data_base
        
    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / x.max() if x.max() > 0 else x

    @staticmethod
    def _intersection(x: np.ndarray,y1: np.ndarray, y2: np.ndarray) -> Optional[Tuple[float, float]]:

        f1 = interp1d(x, y1, bounds_error=False)
        f2 = interp1d(x, y2, bounds_error=False)

        def diff(v: float) -> float:
            return float(f1(v) - f2(v))

        for i in range(len(x) - 1):
            if diff(x[i]) * diff(x[i + 1]) < 0:
                x_star = brentq(diff, x[i], x[i + 1])
                return x_star, float(f1(x_star))

        return None


        

    def mse_threshold(self, target: str, return_curve: bool = False):
        """
        Compute mse threshold
        """

        self.R.regress(self.D_base, self.D, target=target, save_scores=True)

        mse_values = self.R.rules.mean_squared_error.values

        thresholds = np.linspace(mse_values.min(), mse_values.max(), 10)

        total = len(self.R)
        discarded = []
        coverage = []

        for t in thresholds:
            R_t = self.R.filter_by_metric("mean_squared_error", max_value=t)
            print(f"\n\nRULES AFTER FILTERING BY MSE <= {t}: {len(R_t)}\n\n")

            discarded.append(total - len(R_t))
            _, cov = R_t.get_per_index_cov(self.D, return_coverage=True)
            print(f'\nMSE threshold={t}, num_rules={len(R_t)}, coverage={cov}, discarded={total - len(R_t)}\n')
            coverage.append(cov / 100)

        discarded = self._normalize(np.array(discarded))
        coverage = np.array(coverage)
        x = 1 - (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())

        inter = self._intersection(x, discarded, coverage)
        th_opt = inter[0] if inter else None

        if return_curve:
            return th_opt, {
                "x": x,
                "discarded_norm": discarded,
                "coverage": coverage,
                "x_intersection": inter[0] if inter else None,
                "y_intersection": inter[1] if inter else None,
            }

        return th_opt

    

    def accuracy_threshold(self, thresholds: Iterable[float] = np.arange(0.0, 1.1, 0.1), target: Optional[str] = None, return_curve: bool = False):
        
        """
        Compute accuracy threshold.
        """

        if target is None:
            self.R.classify(self.D, save_scores=True)
        else:
            self.R.classify(self.D, target=target, save_scores=True)

        total = len(self.R)
        discarded = []
        coverage = []

        for t in thresholds:
            R_t = self.R.filter_by_metric("accuracy", min_value=t)
            print(f"\n\nRULES AFTER FILTERING BY ACCURACY >= {t}: {len(R_t)}\n\n")
            discarded.append(total - len(R_t))
            _, cov = R_t.get_per_index_cov(self.D, return_coverage=True)
            print(f'\nAccuracy threshold={t}, num_rules={len(R_t)}, coverage={cov}, discarded={total - len(R_t)}\n')
            coverage.append(cov / 100)

        discarded = self._normalize(np.array(discarded))
        coverage = np.array(coverage)
        x = np.array(list(thresholds))
            
        inter = self._intersection(x, discarded, coverage)
        th = inter[0] if inter else None

        if return_curve:
            return th, {
                "x": x,
                "discarded_norm": discarded,
                "coverage": coverage,
                "x_intersection": inter[0] if inter else None,
                "y_intersection": inter[1] if inter else None,
            }

        return th


    def topk_threshold(self,k_values: np.ndarray, return_curve: bool = False, target: Optional[str] = None, target_type: str = 'classification'):
        """
        Compute top-k threshold
        """

        total = len(self.R.rules)
        discarded, coverage = [], []

        k_values = np.sort(k_values)
        k_values = np.clip(k_values, 0, total)

        if target_type == 'regression':
            self.R.regress(self.D_base, self.D, target=target, save_scores=True)
        else:
            self.R.classify(self.D, save_scores=True)


        for k in k_values:
            if k == 0:
                discarded.append(total)
                coverage.append(0)
                continue

            elif k == total:
                discarded.append(0)
                _, cov = self.R.get_per_index_cov(self.D, return_coverage=True)
                coverage.append(cov / 100)
                continue

            if target_type == 'regression':
                R_k = self.R.take_top_k('mean_squared_error', int(k), inplace=False, ascending=True)
                print(f"\n\nRULES TOP k AFTER FILTERING BY <= MSE: {len(R_k)}\n\n")
                print(R_k.rules[['mean_squared_error']])
            else:
                R_k = self.R.take_top_k('accuracy', int(k), inplace=False, ascending=False)
                print(f"\n\nRULES TOP k AFTER FILTERING BY >= ACCURACY: {len(R_k)}\n\n")

            discarded.append(total - len(R_k))
            _, cov = R_k.get_per_index_cov(self.D, return_coverage=True)
            coverage.append(cov / 100)
           
        discarded = self._normalize(np.array(discarded))
        coverage = np.array(coverage)
        x = 1 - (k_values / total)



        order = np.argsort(x)
        x = x[order]
        discarded = discarded[order]
        coverage = coverage[order]

        inter = self._intersection(x, discarded, coverage)

        if inter:
            x_star = np.clip(inter[0], 0, 1)
            th = (1 - x_star) * total  
        else:
            th = None

        if return_curve:
            return th, {
                "x": x,
                "discarded_norm": discarded,
                "coverage": coverage,
                "x_intersection": inter[0] if inter else None,
                "y_intersection": inter[1] if inter else None,
            }

        return th



    def lift_threshold(self, return_curve: bool = False):

        lifts = self.R.rules.lift.values
        if lifts.max() == lifts.min():
            thresholds = np.linspace(0, lifts.max(), 10)
        else: 
            thresholds = np.linspace(lifts.min(), lifts.max(), 10)
        

        total = len(self.R.rules)
        discarded, coverage = [], []

        for l in thresholds:
            R_l = self.R.rules[self.R.rules.lift >= l]

            discarded.append(total - len(R_l))
            idx = rule_utils.count_rules_per_row(self.D, rules=R_l)

            coverage.append(len(idx) / len(self.D))

        discarded = self._normalize(np.array(discarded))
        coverage = np.array(coverage)
        x = thresholds / lifts.max()
        inter = self._intersection(x, discarded, coverage)
        th = inter[0] * lifts.max() if inter else None

        if return_curve:
            return th, {
                "x": x,
                "discarded_norm": discarded,
                "coverage": coverage,
                "x_intersection": inter[0] if inter else None,
                "y_intersection": inter[1] if inter else None,
            }

        return th

    def class_topk(self,rule_set,k_values: np.ndarray,return_curve: bool = False,target_type: str = 'classification'):

        total = len(rule_set.rules)
       
        k_values = np.sort(k_values)
        k_values = np.clip(k_values, 0, total)

        discarded = []
        coverage = []

        for k in k_values:

            if k == 0:
                discarded.append(total)
                coverage.append(0.0)
                continue

            if k >= total:
                discarded.append(0)
                _, cov = rule_set.get_per_index_cov(self.D, return_coverage=True)
                coverage.append(cov / 100)
                continue

            if target_type == 'regression':
                R_k = rule_set.take_top_k('mean_squared_error', int(k), inplace=False, ascending=True)
            else:
                R_k = rule_set.take_top_k('accuracy', int(k), inplace=False, ascending=False)

            discarded.append(total - len(R_k))

            _, cov = R_k.get_per_index_cov(self.D, return_coverage=True)
            coverage.append(cov / 100)

        discarded = np.array(discarded)
        coverage = np.array(coverage)

        discarded_norm = self._normalize(discarded)

        x = 1 - (k_values / total)

        order = np.argsort(x)
        x = x[order]
        discarded_norm = discarded_norm[order]
        coverage = coverage[order]

        inter = self._intersection(x, discarded_norm, coverage)

        if inter:
            x_star = np.clip(inter[0], 0, 1)
            th = (1 - x_star) * total 
        else:
            th = None

        if return_curve:
            return th, {
                "x": x,
                "discarded_norm": discarded_norm,
                "coverage": coverage,
                "x_intersection": inter[0] if inter else None,
                "y_intersection": inter[1] if inter else None,
            }

        return th

   

def per_class_thresholds(
    rule_sets: Dict[str, AprioriRuleSet],
    data_apriori: DataFrame,
    data: DataFrame,
    method: str,
    target: Optional[str] = None,
    return_curve: bool = True,
    **kwargs
):
    thresholds = {}
    curves = {}

    for label, rs in rule_sets.items():
        analyzer = RuleSetMetricAnalyzer(rs, data_apriori, data)

        if method == "accuracy":
            t = analyzer.accuracy_threshold(target=label, **kwargs, return_curve=return_curve)
        elif method == "lift":
            t = analyzer.lift_threshold(**kwargs, return_curve=return_curve)
        elif method == "mean_squared_error":
            t = analyzer.mse_threshold(target=target, **kwargs, return_curve=return_curve)
        else:
            raise ValueError(f"Unknown method: {method}")

        if t is not None:
            if return_curve:
                thr_value, curve = t
                thresholds[label] = thr_value
                curves[label] = curve
            else:
                thresholds[label] = t

       
    if return_curve:
        return thresholds, curves
    else:
        return thresholds


def per_class_topk_thresholds(
    rule_sets: Dict[str, AprioriRuleSet],
    data_apriori: DataFrame,
    data: DataFrame,
    k_values: np.ndarray,
    return_curve: bool = False,
    target: Optional[str] = None,
    target_type: str = 'classification',
):
    

    thresholds = {}
    curves = {}

    for label, rs in rule_sets.items():
        if target_type == 'classification':
            rs.classify(data_apriori, target=label, save_scores=True)
        else:
            rs.regress(data, data_apriori, target=target, save_scores=True)

        analyzer = RuleSetMetricAnalyzer(rs, data_apriori, data)
        print(f"\n\n in per_class_topk_thresholds Computing top-k threshold for class {label}, with {len(rs.rules)} rules\n\n")
        t = analyzer.class_topk(rs, k_values, return_curve=return_curve,target_type=target_type)
        print(f"\n\nTop-k threshold for class {label}: {t}\n\n")
        if t is not None:
            if return_curve:
                best_k, curve = t
                thresholds[label] = best_k
                curves[label] = curve
            else:
                thresholds[label] = t


    if return_curve:
        return thresholds, curves
    else:
        return thresholds



