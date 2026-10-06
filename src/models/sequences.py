import numpy as np
import pandas as pd
import tensorflow as tf
import math

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from typing import List, Tuple, Optional

from src import utils
from pandas.api.types import is_numeric_dtype

DataFramePair = Tuple[pd.DataFrame, pd.DataFrame]
DataFrameQuadruplet = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]


class AbstractSequence(tf.keras.utils.Sequence):
    """Base class for custom tf.keras.Sequence"""

    def __init__(self, seed=utils.SEED, **kwargs):
        self.seed = seed
        self.gen = self._get_random_generator(seed)

    def to_tf_dataset(self):
        return self.dataset_from_sequence(self)

    @staticmethod
    def _get_random_generator(seed) -> np.random.Generator:
        if seed is not None:
            seed = int(seed)
            assert 0 <= seed < 2 ** 32

        return np.random.default_rng(np.random.MT19937(seed=seed))

    @staticmethod
    def dataset_from_sequence(sequence: tf.keras.utils.Sequence,
                              sample_weights=False, prefetch=2):
        def gen():
            for i in range(len(sequence)):
                yield sequence[i]

        out_types = ({'features': tf.float32, 'lhs': tf.int32, 'rhs': tf.int32, 'mask': tf.float32}, tf.float32)

        if sample_weights:
            out_types += (tf.float32,)

        tf_data = tf.data.Dataset.from_generator(gen, output_types=out_types)
        return tf_data.prefetch(prefetch)


class DataSequence(AbstractSequence):
    LOG_TAG = '[DataSequence]'

    def __init__(self, data: pd.DataFrame, rules: Optional[dict], batch_size: int, target_column: str, rule_max_len: int = None, shuffle=False, scaler=None, num_classes=None, num_targets=None, deterministic_rules: bool = False, **kwargs):
        """
        Divides the provided data into mini-batches of the kind dict(x, y, lhs, rhs)
        :param data: the full dataset as a pd.DataFrame.
        :param rules: the rules per index dictionary, obtained by `rules.RuleSet.get_rules_per_index(...)`.
        :param batch_size: number of examples to put in a mini-batch.
        :param target_column: str, the name of the target column.
        :param rule_max_len: maximum size of the left-hand side of a rule. If None, the length will be automatically
                             determined. It is suggested to leave the argument to None.
        :param shuffle: whether to shuffle the data.
        :param scaler: a sklearn.Scaler instance. By default, the StandardScaler is used on numerical data only.
        :param num_classes: number of classes or categories. If None, it will be automatically determined.
        :param num_targets: number of regression variables. If None, it will be automatically determined.
        :param deterministic_rules: if True, for each instance always pick the rule with the
                                    highest confidence among those covering it, instead of
                                    sampling uniformly at random. Ties are broken by the
                                    order rules appear for that instance (stable). Intended
                                    for validation/test sequences, so evaluation metrics are
                                    reproducible instead of depending on a random draw.
                                    Training sequences should keep this False, since the
                                    random sampling acts as a form of regularization/
                                    augmentation across epochs.
        :param kwargs: seed.
        """
        assert not (isinstance(num_classes, int) and isinstance(num_targets, int)), \
            "Should provide only one argument between `num_classes` and `num_targets`, not both!"
        super().__init__(**kwargs)

        self.data = data
        self.index = list(data.index)
        self.rules = rules
        self.batch_size = int(batch_size)
        self.should_shuffle = bool(shuffle)
        self.deterministic_rules = bool(deterministic_rules)

        if isinstance(rule_max_len, int):
            self.rule_max_len = rule_max_len
        elif rules is not None:
            self.rule_max_len = self._find_rule_max_length()
        else:
            self.rule_max_len = -1

        self.target_column = str(target_column)
        self.categorical_columns = [str(col) for col, dtype in data.dtypes.items() if dtype == 'object' and col != self.target_column]
      
        self.feature_columns = [c for c in data.columns if c != self.target_column and (c not in self.categorical_columns)]
        self.num_features = len(self.feature_columns) + len(self.categorical_columns)

        if num_targets is None and num_classes is None:
            target_dtype: np.dtype = self.data[self.target_column].dtype

            if target_dtype.kind == 'f':
                self.num_targets = 1
                self.num_classes = -1
                utils.LOGGER.info(f'{self.LOG_TAG} Target column "{self.target_column}" is numerical.')
            else:
               
                assert target_dtype.kind in ['i', 'u', 'O']
                self.num_targets = -1
                self.num_classes = len(self.data[self.target_column].unique())
                utils.LOGGER.info(f'{self.LOG_TAG} Target column "{self.target_column}" has {self.num_classes} categories.')

        elif isinstance(num_targets, int):
            self.num_targets = num_targets
            self.num_classes = -1
        else:
            self.num_classes = (num_classes or len(self.data[self.target_column].unique()))
            self.num_targets = -1

        self.categories = data[self.categorical_columns]
        self.features = data[self.feature_columns].copy()

        categorical_cols = self.features.select_dtypes(include=['object', 'string', 'category']).columns

        for col in categorical_cols:
            self.features[col] = (self.features[col].astype('category').cat.codes)


        self.targets = data[[self.target_column]]

       

        if not is_numeric_dtype(self.targets[self.target_column]):
            self.targets = self._to_dataframe(OrdinalEncoder().fit_transform(self.targets),columns=[self.target_column])
      

        self.targets = self.targets - self.targets.min()

        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)

        utils.LOGGER.info(f'{self.LOG_TAG} Using "{self.scaler.__class__.__name__}" as scaler.')
        self.features = self._to_dataframe(self.features, columns=self.feature_columns)

       
        encoder = OrdinalEncoder()
        self.categories = self._to_dataframe(encoder.fit_transform(self.categories), columns=self.categorical_columns)
        if self.should_shuffle:
            self.gen.shuffle(self.index)

    def __len__(self):
        return math.ceil(len(self.index) / self.batch_size)

    def __getitem__(self, idx) -> Tuple[dict, list]:
        indices = self._get_indices(idx)
        features = self.features.loc[indices]
        categories = self.categories.loc[indices]
        targets = self.targets.loc[indices]

        if self.num_classes > 2:
            targets = tf.keras.utils.to_categorical(targets, num_classes=self.num_classes)
        if self.rules is None:
            batch = dict()
        else:
            batch = self._get_rules_and_masks(indices)

        batch['features'] = np.concatenate([features, categories], axis=-1)
        return batch, targets

    def _get_indices(self, idx: int):
        start_idx = idx * self.batch_size
        stop_idx = start_idx + self.batch_size
        return self.index[start_idx:stop_idx]

    def _get_rules_and_masks(self, indices: list) -> dict:
        masks = np.zeros(shape=(len(indices), 1), dtype=np.float32)
        batch = dict(lhs=[], rhs=[])

        for k, idx in enumerate(indices):
            if idx in self.rules:
                rules = self.rules[idx]

                if self.deterministic_rules:
                    
                    confidences = [r.get('confidence', 0.0) for r in rules]
                    best_idx = int(np.argmax(confidences))
                    rule = rules[best_idx]
                else:
                    random_idx = self.gen.choice(len(rules))
                    rule = rules[random_idx]

               
                rule = {k: [v for v in values] for k, values in rule.items()
                        if k in ('lhs', 'rhs')}

                for _ in range(len(rule['lhs']), self.rule_max_len):
                    rule['lhs'].append(-1)

                batch['lhs'].append(rule['lhs'])
                batch['rhs'].append(rule['rhs'])
            else:
                masks[k] = 1.0

                batch['lhs'].append([0] * self.rule_max_len)
                batch['rhs'].append([0])

        batch['mask'] = masks
        batch['lhs'] = np.array(batch['lhs'], dtype=np.int32)
        batch['rhs'] = np.array(batch['rhs'], dtype=np.int32)
        return batch

    def get_categorical_columns_mapping(self) -> dict:
        return {c: dict(index=i,
                        num_categories=len(self.categories[c].unique()))
                for i, c in enumerate(self.categorical_columns)}

    def _to_dataframe(self, values, columns: List[str]) -> pd.DataFrame:
        return pd.DataFrame(values, columns=columns, index=self.index)

    def _find_rule_max_length(self) -> int:
        max_len = 0

        for i, rules in self.rules.items():
            for rule in rules:
                if len(rule['lhs']) > max_len:
                    max_len = len(rule['lhs'])

        return max_len

    def on_epoch_end(self):
        if self.should_shuffle:
            self.gen.shuffle(self.index)



    def get_targets(self):
        ys = []
        for i in range(len(self)):
            _, y = self[i]
            ys.append(np.asarray(y))
        return np.concatenate(ys).squeeze()


    @classmethod
    def sequences_from_splits(cls, splits: List[DataFramePair],
                              batch_sizes: tuple, target_column: str, scaler_cls=None,
                              rules=None, seed=None) -> List[Tuple['DataSequence', 'DataSequence']]:
        assert all(len(s) == 2 for s in splits), 'All splits must be pairs!'
        sequences = []

        for train_df, test_df in splits:
            train_seq = cls(train_df, rules=rules, batch_size=batch_sizes[0],
                            target_column=target_column, shuffle=True, seed=seed,
                            scaler=None if scaler_cls is None else scaler_cls())

            if train_seq.num_targets > 0:
                num_targets = train_seq.num_targets
                num_classes = None
            else:
                assert train_seq.num_classes > 0
                num_targets = None
                num_classes = train_seq.num_classes

            test_seq = cls(test_df, rules=rules, batch_size=batch_sizes[1],
                           target_column=target_column, shuffle=False, seed=seed,
                           scaler=train_seq.scaler, num_classes=num_classes,
                           num_targets=num_targets, deterministic_rules=True)

            sequences.append((train_seq, test_seq))

        return sequences


class AprioriSequence(DataSequence):
    def __init__(self, data: pd.DataFrame, apriori_data: pd.DataFrame, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

        self.apriori_data = apriori_data
        assert len(self.data) == len(self.apriori_data)

    def __getitem__(self, idx) -> Tuple[dict, list]:
        batch, labels = super().__getitem__(idx)

        indices = self._get_indices(idx)
        apriori = self.apriori_data.loc[indices]

        batch['apriori'] = apriori
        return batch, labels

    @classmethod
    def sequences_from_splits(cls, splits: List[DataFrameQuadruplet], batch_sizes: tuple,
                              target_column: str, scaler_cls=None, rules=None,
                              seed=None) -> List[Tuple['DataSequence', 'DataSequence']]:
        assert all(len(s) == 4 for s in splits), \
            'All splits must be quadruples: (train, test, apriori_train, apriori_test)!'
        sequences = []

        for train_df, test_df, apriori_train, apriori_test in splits:
            train_seq = cls(train_df, apriori_train, rules=rules, batch_size=batch_sizes[0],
                            target_column=target_column, shuffle=True, seed=seed,
                            scaler=None if scaler_cls is None else scaler_cls())

            if train_seq.num_targets > 0:
                num_targets = train_seq.num_targets
                num_classes = None
            else:
                assert train_seq.num_classes > 0
                num_targets = None
                num_classes = train_seq.num_classes

            test_seq = cls(test_df, apriori_test, rules=rules, batch_size=batch_sizes[1],
                           target_column=target_column, shuffle=False, seed=seed,
                           scaler=train_seq.scaler, num_classes=num_classes,
                           num_targets=num_targets, deterministic_rules=True)

            sequences.append((train_seq, test_seq))

        return sequences














