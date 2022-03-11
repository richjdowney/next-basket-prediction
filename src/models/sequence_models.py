import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Input,
    Bidirectional,
    SpatialDropout1D,
    Dropout,
)
from tensorflow.keras.models import Model
from src.models.next_bask_pred_model import NextBasketPredModel
from src.models.custom_layers import Attention, EncoderLayer


class LSTMModel(NextBasketPredModel, Attention):
    def build(self):
        sequence_input = Input(
            shape=(
                self._max_seq_length,
                self._max_items_in_bask,
            )
        )

        embedded_sequence = Embedding(
            input_dim=self._num_prods + 1,
            input_shape=(self._max_items_in_bask,),
            input_length=self._max_seq_length,
            output_dim=self._d_model,
            mask_zero=True,
            name=self._item_embeddings_layer_name,
        )(sequence_input)

        embedded_sequence *= tf.math.sqrt(tf.cast(self._d_model, tf.float32))

        if self._transformer_encode:
            encoded_sequence = EncoderLayer(
                self._d_model,
                self._num_heads,
                self._dff,
                self._max_seq_length,
                self._max_items_in_bask,
                self._run_pos_encoding,
                rate=0.1,
            )(embedded_sequence, mask=[0])
        else:
            encoded_sequence = embedded_sequence

        if self._basket_pool.lower() == "avg":
            encoded_basket = tf.reduce_mean(encoded_sequence, axis=2)
        elif self._basket_pool.lower() == "dense":
            batch_size = tf.shape(encoded_sequence)[0]
            encoded_basket = tf.reshape(
                encoded_sequence,
                [
                    batch_size,
                    self._max_seq_length,
                    self._max_items_in_bask * self._d_model,
                ],
            )  # (batch_size, max_seq_length, max_items_in_bask*d_model)
            encoded_basket = Dense(self._d_model)(encoded_basket)

        encoded_basket = SpatialDropout1D(0.2)(encoded_basket)

        lstm = Bidirectional(
            LSTM(
                self._lstm_units,
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=True,
            )
        )(encoded_basket)

        attention = Attention()(lstm)

        drop1 = Dropout(0.2)(attention)

        sigmoid = Dense(self._num_prods + 1, activation="sigmoid")(drop1)

        self._model = Model(inputs=[sequence_input], outputs=sigmoid)
