import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers as tfkl
from tensorflow.keras import optimizers as tfo
from typing import List

from src import utils
from src.models.car import CARModel


class RuleRegressor(CARModel):
    """A neural network model conditioned on association rules for regression tasks"""
    def __init__(self, num_features: int, num_columns: int, num_targets: int, embedding_size: int, rule_max_len: int, **kwargs):
        self.num_targets = int(num_targets)

        super().__init__(num_features, num_columns, num_classes=self.num_targets, embedding_size=embedding_size, rule_max_len=rule_max_len, **kwargs)
        del self.num_classes

   
    def compile(self, optimizer_class='adam', loss=None, metrics=None, lr=0.001, **kwargs):
        if loss is None:
            loss = tf.keras.losses.MeanSquaredError()

        super().compile(optimizer_class, loss, metrics, lr, **kwargs)

    @classmethod
    def get_compiled(cls, data: pd.DataFrame, target_column: str, embedding_size: int, num_embedding_columns: int = None, units: list = None, **kwargs):
        opt = kwargs.pop('optimizer', {})
        lr = kwargs.pop('lr', 1e-3)
        compile_args = kwargs.pop('compile', {})
        units = [300, 150, 100, 50] if units is None else units

        model = cls(num_features=len(data.columns) - 1, num_columns=num_embedding_columns or len(data.columns), num_targets=1, embedding_size=embedding_size, units=units, **kwargs)

        model.compile(lr=lr, **opt, **compile_args, metrics=cls.get_default_metrics())
        return model

    @classmethod
    def get_default_metrics(cls, *args, **kwargs) -> List[tf.keras.metrics.Metric]:
        return [tf.keras.metrics.R2Score(), tf.keras.metrics.MeanSquaredError()]

    def output_layer(self, layer: tfkl.Layer, name='values', **kwargs):
        return tfkl.Dense(units=self.num_targets, activation=kwargs.pop('activation', 'linear'), name=name, **kwargs)(layer)


class Regressor(RuleRegressor):
    def __init__(self, num_features: int, num_targets: int, **kwargs):
        [kwargs.pop(k, None) for k in ['num_columns', 'embedding_size', 'rule_max_len']]

        super().__init__(num_features, num_columns=0, num_targets=num_targets, embedding_size=0, rule_max_len=0, **kwargs)

    def architecture(self, units: List[int], activation='relu', categorical_columns: dict = None, category_embedding_size=2, category_max_one_hot_size: int = 5, dropout=0.0, trainable_embeddings=False, **kwargs) -> tuple:
        assert len(units) > 1

        output_args = kwargs.pop('output', {})
        apply_dropout = dropout > 0.0

        features = tf.keras.Input(shape=(self.num_features,), name='features')

        if isinstance(categorical_columns, dict):
            x = self.encode_categorical_features(features, categorical_columns, category_max_one_hot_size, category_embedding_size, trainable_embeddings=bool(trainable_embeddings))
        else:
            x = features

        for i, num_units in enumerate(units):
            x = tfkl.Dense(units=num_units, activation=activation, **kwargs, name=f'dense-{i}')(x)

            if apply_dropout:
                x = tfkl.Dropout(rate=dropout, seed=utils.SEED, name=f'dropout-{i}')(x)

        outputs = self.output_layer(layer=x, **output_args)
        return dict(features=features), outputs

    @classmethod
    def get_compiled(cls, data: pd.DataFrame, target_column: str, units: list = None, **kwargs):
        opt = kwargs.pop('optimizer', {})
        lr = kwargs.pop('lr', 1e-3)
        compile_args = kwargs.pop('compile', {})
        units = [300, 150, 100, 50] if units is None else units

        model = cls(num_features=len(data.columns) - 1, num_targets=1, units=units, **kwargs)

        model.compile(lr=lr, **opt, **compile_args, metrics=cls.get_default_metrics())
        return model
