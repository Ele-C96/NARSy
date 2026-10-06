import json
from pyexpat import features
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path

from tensorflow.keras import layers as tfkl
from tensorflow.keras import optimizers as tfo
from tensorflow.keras import metrics as tfkm

from src import utils
from src.models import AbstractSequence, RuleConditioning
from src.models.layers import ConcatConditioning, RulePreprocessing


def _get_training_device() -> str:
    return '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'


class CARModel(tf.keras.Model):
    """A neural network model conditioned on association rules:
        - Classification Association Rule (CAR) Model"""

    def __init__(self, num_features: int, num_columns: int, num_classes: int,
                 embedding_size: int, rule_max_len: int, **kwargs):
        """
        :param num_features: number of input features. Typically, the #columns of the dataset minus the target
                             variable(s).
        :param num_columns: total number of columns of the dataset.
        :param num_classes: number of unique classes of the target variable (assuming a classification problem).
        :param embedding_size: dimensionality of a single embedding resulting when encoding the rules.
        :param rule_max_len: maximum size of the left-hand side of a rule.
        :param kwargs: arguments passed to the architecture() method.
        """


        self._base_model_initialized = True
        self.num_features = int(num_features)
        self.num_columns = int(num_columns)
        self.num_classes = int(num_classes)
        self.rule_length = int(rule_max_len)
        self.embedding_size = int(embedding_size)
        self.training_device = _get_training_device()

        name = kwargs.get('name', self.__class__.__name__)
        self.embed: tfkl.Embedding = None

        inputs, outputs = self.architecture(**kwargs)
        super().__init__(inputs, outputs, name=name)

        self.history = dict()
        self.best_weights = None

    
    def compile(self, optimizer_class='adam', loss=None, metrics=None,
            lr=0.001, **kwargs):
        if isinstance(optimizer_class, str):
            optimizer_class = dict(
                sgd=tf.keras.optimizers.SGD,
                adam=tf.keras.optimizers.Adam,
                nadam=tf.keras.optimizers.Nadam,
                rmsprop=tf.keras.optimizers.RMSprop
            )[optimizer_class.lower()]

        optimizer = optimizer_class(learning_rate=lr, clipnorm=1.0, **kwargs)

        if loss is None:
        
            loss = 'categorical_crossentropy' \
                if self.num_classes > 2 else 'binary_crossentropy'

        self.custom_metrics = [
        tf.keras.metrics.get(m)
        for m in (metrics or [])
    ]

        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=self.custom_metrics
        )

    def architecture(self, units: List[int], activation='relu', categorical_columns: dict = None, category_embedding_size=2,
                     category_max_one_hot_size: int = 5, concat_conditioning=False, dropout=0.0, condition_all_layers=False, **kwargs) -> tuple:
        """
        :param units: list of integers denoting the number of layers and neurons to use in each layer. E.g., [128, 64],
                      yields a two-layer network with 128 units on the first layer, and 64 for the second.
        :param activation: activation function, defaults to "relu".
        :param categorical_columns: dictionary [str, dict(num_categories, index)] mapping the names of the categorical
                                    columns of the dataset.
        :param category_embedding_size: dimensionality of the embedding for a given category.
        :param category_max_one_hot_size: size after which, one-hot encoded categories will be, instead, embedded into
                                          dense vectors of dimensionality `category_embedding_size`.
        :param concat_conditioning: bool, whether to use concatenation instead of affine-conditioning. False by default.
        :param dropout: dropout rate. By default, no dropout regularization is used.
        :param condition_all_layers: whether to apply the rule-conditioning mechanism on all hidden layers. By default,
                                     the conditioning is applied only on the first hidden layer.
        :param kwargs: additional arguments to pass to tf.keras.Layer. For example, "kernel_initializer" or
                       "kernel_regularizer" can be provided.
        :param trainable_embeddings: bool, whether to train the embedding's lookup tables of both rules and categories.
                                     By default, the embeddings are fixed.
        """
        assert len(units) > 0

        output_args = kwargs.pop('output', {})
        apply_dropout = dropout > 0.0
        trainable_embeddings = kwargs.pop('trainable_embeddings', False)

        features = tf.keras.Input(shape=(self.num_features,), name='features')
        rule_lhs = tf.keras.Input(shape=(self.rule_length,), name='lhs', dtype=tf.int32)
        rule_rhs = tf.keras.Input(shape=(1,), name='rhs', dtype=tf.int32)
        mask = tf.keras.Input(shape=(1,), name='rule_mask')

        if isinstance(categorical_columns, dict):
            x = self.encode_categorical_features(features, categorical_columns, category_max_one_hot_size, category_embedding_size, trainable_embeddings=trainable_embeddings)
        else:
            x = features

        lhs, rhs = self._replace_padding_value_in_rules(rule_lhs, rule_rhs)

       
        self.embed = tfkl.Embedding(input_dim=self.num_columns + 2,output_dim=self.embedding_size,name='Embedder')

        lhs = self.embed(lhs)
        lhs = RulePreprocessing(rule_length=self.rule_length,name='lhs_preproc')([rule_lhs, lhs])

        rhs = self.embed(rhs)
        rhs = tf.keras.layers.Reshape((self.embedding_size,))(rhs)
        rhs.set_shape(shape=(None, self.embedding_size))

        for i, num_units in enumerate(units):
            x = tfkl.Dense(units=num_units, activation=activation, kernel_initializer=tf.keras.initializers.HeUniform(
        seed=utils.SEED), **kwargs, name=f'dense-{i}')(x)

            if condition_all_layers or i == 0:
                if concat_conditioning:
                    x = ConcatConditioning(name=f'concat_cond-{i}')([x, lhs, rhs], mask)
                else:
                    x = RuleConditioning(name=f'conditioning-{i}')([x, lhs, rhs], mask)

            if apply_dropout and i > 0:
                x = tfkl.Dropout(rate=dropout, seed=utils.SEED, name=f'dropout-{i}')(x)

        outputs = self.output_layer(layer=x, **output_args)

        return dict(features=features, lhs=rule_lhs, rhs=rule_rhs, mask=mask), outputs

    def output_layer(self, layer: tfkl.Layer, name='classes', **kwargs):
        if self.num_classes <= 2:
            return tfkl.Dense(
                units=1,
                activation='sigmoid',
                dtype='float32',
                name=name,
                **kwargs
            )(layer)

        return tfkl.Dense(
            units=self.num_classes,
            activation='softmax',
            dtype='float32',
            name=name,
            **kwargs
        )(layer)

    def encode_categorical_features(self, features, categorical_columns: dict, max_one_hot_size, embedding_size, trainable_embeddings=False) -> tfkl.Layer:
        if not categorical_columns:
            return features

        num_columns = len(list(categorical_columns.keys()))
        num_numeric = self.num_features - num_columns


        split_layer = tf.keras.layers.Lambda(
            lambda x: tf.split(
                x,
                [num_numeric, num_columns],
                axis=-1
            )
        )

        features_, categories = split_layer(features)

        categorical_layers = []

        for column, specs in categorical_columns.items():
            if specs['num_categories'] <= max_one_hot_size:
                layer = tfkl.CategoryEncoding(num_tokens=specs['num_categories'],
                                              output_mode='one_hot', name=f'OHE-{column}')

                layer = layer(categories[:, specs['index'], tf.newaxis])
            else:
                layer = tfkl.Embedding(input_dim=specs['num_categories'],
                                       output_dim=embedding_size,
                                       embeddings_initializer=tf.keras.initializers.Orthogonal(seed=utils.SEED),
                                       name=f'Embedding-{column}')

                if not trainable_embeddings:
                    layer.trainable = False

                layer = layer(categories[:, specs['index']])

            categorical_layers.append(layer)

        concat_layer = tf.keras.layers.Concatenate(axis=-1)

        categories = concat_layer(categorical_layers)
        final_concat = tf.keras.layers.Concatenate(axis=-1)

        return final_concat([features_, categories])

    @tf.function
    def train_step(self, batch):

        x, y_true, sample_weight = self._unpack(batch)

        with tf.GradientTape() as tape:

            y_pred = self(x, training=True)

            loss = self.compute_loss(
                x=x,
                y=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
            )

            loss = tf.reduce_mean(loss)

        weight_norm, global_norm = self.apply_gradients(tape, loss)

        for metric in self.custom_metrics:
            metric.update_state(y_true, y_pred)

        debug = {
            "loss": loss,
            "grad-norm": global_norm,
            "weight-norm": weight_norm,
            "reg-losses": tf.reduce_sum(self.losses)
        }

        for metric in self.custom_metrics:
            debug[metric.name] = metric.result()

        return debug
    
    @tf.function
    def test_step(self, batch):

        x, y_true, sample_weight = self._unpack(batch)

        y_pred = self(x, training=False)

        loss = self.compute_loss(
            x=x,
            y=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )

        loss = tf.reduce_mean(loss)

        for metric in self.custom_metrics:
            metric.update_state(y_true, y_pred)

        debug = {
            "loss": loss
        }

        for metric in self.custom_metrics:
            debug[metric.name] = metric.result()

        return debug


    def predict(self, *args, **kwargs):
        history = self.history
        output = super().predict(*args, **kwargs)

        self.history = history
        return output

    def train(self, train_sequence: AbstractSequence, epochs: int, digits=4,
              validation_sequence: AbstractSequence = None, monitor='auto',
              mode='max', load_best_weights=True, verbose=True, simple=True):
        """
        Training loop.
        :param train_sequence: sequence providing training data in batches.
        :param epochs: number of training epochs.
        :param digits: number of rounding digits for the metric logging.
        :param validation_sequence: sequence providing validation data in batches.
        :param monitor: target metric to monitor, used to determine the best weights.
        :param mode: set "min" if lower is better, or "max" if higher is better. The reference metric is the one
                     specified by "monitor".
        :param load_best_weights: whether to load the best weights during each epoch at the end of the training.
        :param verbose: logging verbosity.
        :param simple: bool, simplified logging.
        """
       
        assert epochs > 0
        mode = mode.lower()

        if isinstance(monitor, str):
            save_best = True
            best_metric = np.inf if mode == 'min' else -np.inf

            monitor = monitor.lower()
            if monitor == 'auto':
                mode = 'max'

                if isinstance(validation_sequence, AbstractSequence):
                    monitor = 'val_auc'
                else:
                    monitor = 'auc'
        else:
            save_best = False

        if simple:
            main_bar = tqdm(range(epochs), postfix='Epoch', disable=not verbose)
        else:
            main_bar = range(epochs)
        
        for i in main_bar:
                     
            for metric in self.custom_metrics:
                metric.reset_state()

            bar = tqdm(train_sequence, desc="", disable=(not verbose) or simple,
                       postfix=f'Epoch: {i + 1}/{epochs}')
            for batch in bar:
                info = self.train_step(batch)
                string = ""
                for k, v in info.items():
                    self.history.setdefault(k, [])
                    if isinstance(v, dict):
                        v = list(v.values())[0]

                    if tf.is_tensor(v):
                        v = v.numpy()

                    self.history[k].append(float(v))

                    string += f'{k}: {round(float(v), digits)},'

                if not simple:
                    bar.set_description(string, refresh=True)

            if validation_sequence is not None:
                for metric in self.custom_metrics:
                    metric.reset_state()
                valid_info = self.validation(validation_sequence)
                string = ''

                for k, v in valid_info.items():
                    k = f'val_{k}'

                    self.history.setdefault(k, [])
                    self.history[k].append(v)

                    string += f'{k}: {round(v, digits)},'

                if simple:
                    main_bar.set_description(string, refresh=True)
                else:
                    bar.set_description(string, refresh=True)

            if save_best:
                
                assert monitor in self.history, \
                    f"Available metrics names: {list(self.history.keys())}"
                metric_value = self.history[monitor][-1]

                if mode == 'min':
                    if metric_value < best_metric:
                        self.best_weights = self.get_weights()
                        best_metric = metric_value
                else:
                    if metric_value > best_metric:
                        self.best_weights = self.get_weights()
                        best_metric = metric_value

            train_sequence.on_epoch_end()

        if save_best and load_best_weights:
            self.set_weights(self.best_weights)
            return best_metric

    def validation(self, sequence: AbstractSequence) -> dict:
        stats = dict()

   
        for batch in sequence:
            info = self.test_step(batch)

            for k, v in info.items():

                if isinstance(v, dict):

                    for sub_k, sub_v in v.items():

                        metric_name = f"{k}_{sub_k}"

                        if metric_name not in stats:
                            stats[metric_name] = []

                        if tf.is_tensor(sub_v):
                            sub_v = sub_v.numpy()

                        stats[metric_name].append(float(sub_v))

                else:

                    if k not in stats:
                        stats[k] = []

                    if tf.is_tensor(v):
                        v = v.numpy()

                    stats[k].append(float(v))

        return {
            k: float(np.mean(v))
            for k, v in stats.items()
        }
    def apply_gradients(self, tape, loss):

        variables = self.trainable_variables

        grads = tape.gradient(loss, variables)

        grads = [
            tf.where(
                tf.math.is_finite(g),
                g,
                tf.zeros_like(g)
            ) if g is not None else None
            for g in grads
        ]

        self.optimizer.apply_gradients(
            zip(grads, variables)
        )

        return (
            utils.tf_global_norm(variables),
            utils.tf_global_norm(grads)
        )

    

    @classmethod
    def get_compiled(cls, data: pd.DataFrame, target_column: str,
                     embedding_size: int, num_embedding_columns: int = None,
                     units: list = None, curve='ROC', **kwargs) -> 'CARModel':
        """
        Function that returns an instantiated and compiled model, ready to be trained.
        :param data: the training data-frame.
        :param target_column: str, the name of the target column.
        :param embedding_size: see __init__.
        :param num_embedding_columns: the number of columns of the transformed dataset.
        :param units: see architecture().
        :param curve: can be "ROC" or "PR". By default, the tracked metric is the area of the ROC.
        :param kwargs: see architecture().
        :return: an instance of CARModel.
        """
        opt = kwargs.pop('optimizer', {})
        lr = kwargs.pop('lr', 1e-3)
        compile_args = kwargs.pop('compile', {})
        units = [300, 150, 100, 50] if units is None else units

        num_classes = len(data[target_column].unique())
        model = cls(num_features=len(data.columns) - 1, num_columns=num_embedding_columns or len(data.columns), num_classes=num_classes, embedding_size=embedding_size, units=units, **kwargs)

        
        model.compile(lr=lr, metrics=cls.get_default_metrics(num_classes, curve=curve),**opt,**compile_args)
        return model
    
    @classmethod
    def get_default_metrics(cls, num_classes: int, *args, curve='ROC', **kwargs) -> List[tfkm.Metric]:

        metrics = [
            tfkm.BinaryAccuracy(name='binary_accuracy')
            if num_classes <= 2
            else tfkm.CategoricalAccuracy(name='categorical_accuracy'),

            tfkm.AUC(name='auc', curve=str(curve).upper()),
            tfkm.Precision(thresholds=0.5 if num_classes <= 2 else None, name='precision'),
            tfkm.Recall(thresholds=0.5 if num_classes <= 2 else None, name='recall'),
            tfkm.F1Score(average='weighted', threshold=0.5 if num_classes <= 2 else None, name='f1_score')
        ]

        if num_classes > 3:
            metrics.append(tfkm.TopKCategoricalAccuracy(k=num_classes - 2, name=f'top_{num_classes - 2}_acc'))

        return metrics



    def save_history(self, path: str, indent=2):
        path = Path(path)

        if not path.parent.exists():
            print(f'Creating folder "{path.parent.name}"')
            path.parent.mkdir(parents=True, exist_ok=True)

        json.dump({k: [float(x) for x in v] for k, v in self.history.items()}, fp=open(path, 'w'), indent=int(indent))

    def save_model(self, weights_name: str, history_name: str = None, base_dir='weights'):
        """Saves the model's (best) weights and optionally the training history as json"""
        weights_path = Path(base_dir) / weights_name
        weights_path.parent.mkdir(exist_ok=True)

        self.save_weights(weights_path)

        if isinstance(history_name, str):
            self.save_history(path=str(Path(base_dir) / history_name))

    @classmethod
    def load(cls, path: str) -> 'CARModel':
        """Instantiate a new model, loading the weights saved at the given path."""
        assert Path(path).exists(), f'Path "{path}" does not exist!'
        raise NotImplementedError

    def _replace_padding_value_in_rules(self, rule_lhs, rule_rhs) -> tuple:
        """Replaces padding value "-1" in rules before embedding them"""
        replace_padding = tf.keras.layers.Lambda(
            lambda x: tf.where(x == -1,self.num_columns + 1,x)
        )

        lhs = replace_padding(rule_lhs)
        rhs = replace_padding(rule_rhs)
        return lhs, rhs

   
    @staticmethod
    def _unpack(batch) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        if isinstance(batch, tuple) and len(batch) == 1:
            batch = batch[0]

        if len(batch) == 3:
            x, targets, sample_weight = batch

        else:
            x, targets = batch

            sample_weight = tf.ones(
                (tf.shape(targets)[0],),
                dtype=tf.float32
            )

        targets = tf.cast(targets, tf.float32)
        sample_weight = tf.cast(sample_weight, tf.float32)

        return x, targets, sample_weight