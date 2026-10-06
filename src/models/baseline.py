import pandas as pd
import tensorflow as tf

from typing import List
from tensorflow.keras import layers as tfkl

from src import utils
from src.models import CARModel


class Classifier(CARModel):
    def __init__(self, num_features: int, num_classes: int, **kwargs):
        [kwargs.pop(k, None) for k in ['num_columns', 'embedding_size', 'rule_max_len']]

        super().__init__(num_features, num_columns=0, num_classes=num_classes, embedding_size=0, rule_max_len=0, **kwargs)

    def architecture(self, units: List[int], activation='relu', categorical_columns: dict = None, category_embedding_size=2, category_max_one_hot_size: int = 5, trainable_embeddings=False, dropout=0.0,**kwargs) -> tuple:
        assert len(units) > 0

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
    def get_compiled(cls, data: pd.DataFrame, target_column: str, units: list = None, curve='ROC', **kwargs):
        opt = kwargs.pop('optimizer', {})
        lr = kwargs.pop('lr', 1e-3)
        compile_args = kwargs.pop('compile', {})
        units = [300, 150, 100, 50] if units is None else units

        num_classes = len(data[target_column].unique())
        model = cls(num_features=len(data.columns) - 1, num_classes=num_classes, units=units, **kwargs)

        model.compile(lr=lr, **opt, **compile_args, metrics=cls.get_default_metrics(num_classes, curve=curve))
        return model
