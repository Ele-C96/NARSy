"""
src/mining.py

Extraction of frequent item sets and association rules (Apriori),
both globally and separately for each target class.
"""

from mlxtend.frequent_patterns import apriori, association_rules

from src.rules import RuleSet


def mine_rules(df_encoded, min_support: float, min_confidence: float) -> RuleSet:
    """Extract association rules from a one-hot encoded dataframe."""
    itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    return RuleSet(rules)


def mine_rules_per_class(df_encoded, target_columns, min_support: float, min_confidence: float):
    """
    Extract rules separately for each target class (one-vs-rest):
    filter the rows of the class, remove the other target columns from the dataframe,
    mine the rules and keep only those whose rhs concerns the current class.

    Returns a dict {target_column_name: RuleSet}.
    """
    rule_sets = {}

    for target_col in target_columns:
        df_class = df_encoded[df_encoded[target_col] == 1].copy()
        df_class = df_class.drop(columns=[c for c in target_columns if c != target_col])

        rule_set = mine_rules(df_class, min_support=min_support, min_confidence=min_confidence)
        rule_set.filter_by_rhs(attribute=target_col)
        rule_sets[target_col] = rule_set

    return rule_sets