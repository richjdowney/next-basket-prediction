import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Dense,
    Layer,
    Dropout,
    LayerNormalization,
)
import numpy as np


class Attention(Layer):
    def __init__(self, add_bias=True, mask=None, return_attention_weights=False):
        super(Attention, self).__init__()
        self.add_bias = add_bias
        self.init = tf.keras.initializers.get("glorot_uniform")
        self.mask = mask
        self.return_attention_weights = return_attention_weights

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer=self.init,
        )

        if self.add_bias:
            self.b = self.add_weight(shape=(input_shape[1],), initializer="zero")

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


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, max_seq_length):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, max_items_in_bask, depth)
        """
        x = tf.reshape(
            x, (batch_size, self.max_seq_length, -1, self.num_heads, self.depth)
        )
        return tf.transpose(x, perm=[0, 1, 3, 2, 4])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len_q, max_items_in_bask_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, max_items_in_bask_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, max_items_in_bask_v, d_model)

        q = self.split_heads(
            q, batch_size
        )  # (batch_size, num_heads, seq_len_q, max_items_in_bask_q, depth)
        k = self.split_heads(
            k, batch_size
        )  # (batch_size, num_heads, seq_len_k, max_items_in_bask_k, depth)
        v = self.split_heads(
            v, batch_size
        )  # (batch_size, num_heads, seq_len_v, max_items_in_bask_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 1, 3, 2, 4]
        )  # (batch_size, seq_len_q, max_items_in_basket_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, self.max_seq_length, -1, self.d_model)
        )  # (batch_size, seq_len_q, max_items_in_bask_q, d_model)

        output = self.dense(
            concat_attention
        )  # (batch_size, seq_len_q, max_items_in_bask_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            Dense(dff, activation="relu"),  # (batch_size, max_items_in_bask, dff)
            Dense(d_model),  # (batch_size, max_items_in_bask, d_model)
        ]
    )


# ----- positional encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class EncoderLayer(Layer):
    def __init__(
        self, d_model, num_heads, dff, max_seq_length, max_items_in_bask, run_pos_encoding, rate=0.1
    ):

        super(EncoderLayer, self).__init__()

        self.max_items_in_bask = max_items_in_bask
        self.run_pos_encoding = run_pos_encoding
        self.pos_encoding = positional_encoding(max_items_in_bask, d_model)

        self.mha = MultiHeadAttention(d_model, num_heads, max_seq_length)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, mask):

        if self.run_pos_encoding:
            x += self.pos_encoding[:, :self.max_items_in_bask, :]

        x = self.dropout1(x)

        attn_output, _ = self.mha(
            x, x, x, mask
        )  # (batch_size, input_seq_len, max_items_in_bask, d_model)
        attn_output = self.dropout2(attn_output)
        out1 = self.layernorm1(
            x + attn_output
        )  # (batch_size, input_seq_len, max_items_in_bask, d_model)

        ffn_output = self.ffn(
            out1
        )  # (batch_size, input_seq_len, max_items_in_bask, d_model)
        ffn_output = self.dropout3(ffn_output)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, max_items_in_bask, d_model)

        return out2
