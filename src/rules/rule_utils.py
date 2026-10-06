import json
import numpy as np
import pandas as pd

from sklearn.base import defaultdict
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Union, Tuple, List
from sklearn.preprocessing import MinMaxScaler
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.dependencies import load_dependencies_file


def get_rules_per_index(df: pd.DataFrame, file_path: str, discard_key_dependencies=True, encode_to_integers=True) -> Dict[int, list]:
    if encode_to_integers:
        mapper = {col: i for i, col in enumerate(df.columns)}
    else:
        mapper = {col: col for col in df.columns}

    dependencies = load_dependencies_file(file_path)
    rules_per_index = dict()
    num_columns = len(df.columns)

    for lhs, rhs in tqdm(dependencies.items()):
        if discard_key_dependencies:
            if len(lhs) + len(rhs) == num_columns:
                continue

        rule = [attr.name for attr in lhs | rhs]

        for attr in rhs:
            value = dict(lhs=[mapper[a.name] for a in lhs],
                         rhs=[mapper[attr.name]])

            for index in df.groupby(by=rule).groups.values():
                index = int(index[0])
                rules_per_index.setdefault(index, [])
                rules_per_index[index].append(value)

    return rules_per_index

def get_apriori_rules_per_index(df: pd.DataFrame, rules_path: str = None, rules: Union[dict, pd.DataFrame] = None, top_k: Union[int, float] = None) -> Dict[int, list]:
    if isinstance(rules_path, str):
        rules: dict = json.load(fp=open(rules_path, 'r'))
    else:
        assert isinstance(rules, (dict, pd.DataFrame)), 'Provide rules or path!'

    mapper = {c: i for i, c in enumerate(df.columns)}

    num_rules = len(rules['antecedents'])
    if top_k is None:
        top_k = num_rules

    elif isinstance(top_k, float):
        top_k = int(float(top_k) * num_rules)
    else:
        assert isinstance(top_k, int)
        top_k = int(top_k)

    print(f'TopK: {top_k}, #rules: {num_rules}')
    rules = {k: v[:top_k] for k, v in rules.items()}

    rules_per_index = {}

    masks = {col: df[col] == 1 for col in df.columns}



    for lhs, rhs_list in tqdm(zip(rules['antecedents'], rules['consequents']), total=num_rules):
        for rhs in rhs_list:
            rule = dict(lhs=[], rhs=[mapper[rhs]])
            mask = None

            for attr in lhs:
                rule['lhs'].append(mapper[attr])
                mask = masks[attr].copy() if mask is None else (mask & masks[attr])

            if mask is None or not mask.any():
                continue

            for index in df[mask].index:
                index = int(index)

                rules_per_index.setdefault(index, [])
                rules_per_index[index].append(rule)


    rule_counts = [len(v) for v in rules_per_index.values()]

    print("\n===== RULE STATISTICS =====")
    print(f"Covered rows: {len(rule_counts)}")
    print(f"Average rules per row: {np.mean(rule_counts):.2f}")
    print(f"Median rules per row: {np.median(rule_counts):.2f}")
    print(f"Max rules per row: {np.max(rule_counts)}")
    print(f"Min rules per row: {np.min(rule_counts)}")
    print("===========================\n")

    
    return rules_per_index


def count_rules_per_row(df: pd.DataFrame, rules_path: str = None, rules: Union[dict, pd.DataFrame] = None, top_k: Union[int, float] = None) -> Dict[int, list]:
    if isinstance(rules_path, str):
        rules: dict = json.load(fp=open(rules_path, 'r'))
    else:
        assert isinstance(rules, (dict, pd.DataFrame)), 'Provide rules or path!'


    num_rules = len(rules['antecedents'])
    if top_k is None:
        top_k = num_rules

    elif isinstance(top_k, float):
        top_k = int(float(top_k) * num_rules)
    else:
        assert isinstance(top_k, int)
        top_k = int(top_k)

    print(f'TopK: {top_k}, #rules: {num_rules}')
    rules = {k: v[:top_k] for k, v in rules.items()}

    masks = {col: df[col] == 1 for col in df.columns}

    covered = set()

    for index in df.index:

        for lhs, rhs_list in zip(rules['antecedents'], rules['consequents']):

            for rhs in rhs_list:

                if not masks[rhs][index]:
                    continue

                if all(masks[attr][index] for attr in lhs):
                    covered.add(index)
                    break   

            if index in covered:
                break      


    print("\n2. [COUNT_RULES] End")
    print(f"DF total rows: {len(df)}")
    print(f"Indices covered by rules: {len(covered)}")
    print(f"Coverage: {len(covered) / len(df)}\n")
    return covered



def format_and_save_rules(association_rules, save_path: str, round_digits=2):
    for k in ['support', 'confidence', 'lift']:
        association_rules[k] = association_rules[k].round(int(round_digits))

    rules_dict = {k: list(association_rules[k]) for k in association_rules.keys()}
    rules_dict['antecedents'] = [list(v) for v in rules_dict['antecedents']]
    rules_dict['consequents'] = [list(v) for v in rules_dict['consequents']]

    json.dump(rules_dict, fp=open(save_path, 'w'))


def pareto_frontiers(rules_or_path: Union[str, pd.DataFrame], criteria: list = None, inplace=True, return_rules=False) -> Union[np.ndarray, Tuple[pd.DataFrame, np.ndarray]]:
    if isinstance(rules_or_path, str):
        path = Path(rules_or_path)
        assert path.exists(), f"No file at provided path: \"{path}\"!"

        rules = json.load(fp=open(path))
        rules = pd.DataFrame(rules)

        return_rules = True
    else:
        assert isinstance(rules_or_path, pd.DataFrame)
        rules = rules_or_path

    if criteria is None or (isinstance(criteria, (list, tuple)) and len(criteria) == 0):
        criteria = [c for c in rules.columns if c not in ['antecedents', 'consequents']]
        assert len(criteria) > 0, "No criteria were saved along the rules..."

    scaler = MinMaxScaler()
    variables = scaler.fit_transform(rules[criteria])

    nds = NonDominatedSorting()
    fronts = nds.do(variables, only_non_dominated_front=False)

    ranks = np.empty(len(rules), dtype=int)

    for i, front in enumerate(fronts):
        ranks[front] = i + 1

    if inplace:
        rules['Pareto_rank'] = ranks

    if return_rules:
        return rules, ranks

    return ranks


def split_rules_by_frontier(rules: pd.DataFrame, ranks: np.ndarray) -> List[pd.DataFrame]:
    rules_list = []

    for frontier_id in np.unique(ranks):
        mask = ranks == frontier_id
        rules_list.append(rules[mask])

    return rules_list









