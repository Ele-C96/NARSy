import tensorflow as tf

from tensorflow.keras import layers as tfkl


class ConcatConditioning(tfkl.Layer):

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.kwargs = kwargs
        self.linear: Linear = None

    def call(self, inputs, mask: tf.Tensor, **kwargs):
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == 3

        # concatenate inputs
        x, lhs, rhs = inputs
        mask = tf.cast(mask, dtype=tf.bool)

        concat = tf.concat([x, lhs, rhs], axis=-1)
        x_proj = self.linear(x, **kwargs)

        z = tf.where(mask, x=x_proj, y=concat)
        return z

    def build(self, input_shape: list):
        x_shape, lhs_shape, rhs_shape = input_shape
        num_units = x_shape[-1] + lhs_shape[-1] + rhs_shape[-1]

        self.linear = Linear(units=num_units, **self.kwargs)


class RuleConditioning(tfkl.Layer):
    """Conditioning: scale by rule's lhs and bias by rule's rhs"""
    def __init__(self, scale_activation='linear', bias_activation='linear',
                 name=None, **kwargs):
        super().__init__(name=name)
        self.kwargs = kwargs

        self.scale_activation = scale_activation
        self.bias_activation = bias_activation

        self.dense_scale: tfkl.Dense = None
        self.dense_bias: tfkl.Dense = None

        self.multiply = tfkl.Multiply()
        self.add = tfkl.Add()

    def build(self, input_shape: list):
        shape, _, _ = input_shape

        self.dense_scale = tfkl.Dense(units=shape[-1],activation=self.scale_activation,**self.kwargs)
        self.dense_bias = tfkl.Dense(units=shape[-1], activation=self.bias_activation,**self.kwargs)

    def call(self, inputs, mask: tf.Tensor, **kwargs):
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == 3

    
        x, lhs, rhs = inputs
        mask = tf.cast(mask, dtype=tf.bool)

        scale = self.dense_scale(lhs, **kwargs)
        scale = tf.where(mask, x=1.0, y=scale)

        bias = self.dense_bias(rhs, **kwargs)
        bias = tf.where(mask, x=0.0, y=bias)

        z = self.multiply([x, scale])
        z = self.add([z, bias])
        return z


class Linear(tfkl.Dense):
    """Linear combination layer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='linear', **kwargs)


class RulePreprocessing(tfkl.Layer):
    def __init__(self, *args, rule_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.rule_length = int(rule_length)

    def call(self, inputs, **kwargs):
        rule_lhs, emb_lhs = inputs
        condition = rule_lhs == (self.rule_length + 1)

        lhs = tf.where(condition[..., tf.newaxis], x=0.0, y=emb_lhs)
        lhs = tf.reduce_sum(lhs, axis=1)
        lhs = lhs / tf.reduce_sum(tf.cast(tf.logical_not(condition), dtype=tf.float32), keepdims=True, axis=-1)
        return lhs

