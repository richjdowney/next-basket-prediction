import tensorflow as tf
from tensorflow.keras.layers import (Layer)
import keras.backend as K


class Attention(Layer):
    def __init__(self, add_bias=True, mask=None, return_attention_weights=False):
        super(Attention, self).__init__()
        self.add_bias = add_bias
        self.init = tf.keras.initializers.get('glorot_uniform')
        self.mask = mask
        self.return_attention_weights = return_attention_weights

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],), initializer=self.init,)

        if self.add_bias:
            self.b = self.add_weight(shape=(input_shape[1],), initializer='zero')

    def call(self, x):

        if self.add_bias:
            eij = tf.tensordot(x, self.W, 1) + self.b
        else:
            eij = tf.tensordot(x, self.W, 1)

        eij = tf.tanh(eij)
        a = K.softmax(eij)

        # Set the softmax output to zero for masked sequence values
        if self.mask is not None:
            a *= self.mask

        a = K.expand_dims(a, axis=-1)

        # Add small epsilon to a to avoid return NaN values
        a /= K.sum(a, axis=1, keepdims=True) + K.epsilon()

        weighted_output = x * a

        result = K.sum(weighted_output, axis=1)

        if self.return_attention_weights:
            return [result, a]

        return result
