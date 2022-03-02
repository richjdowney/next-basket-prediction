import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Input,
    Bidirectional,
    SpatialDropout1D,
)
from tensorflow.keras.models import Model
from src.models.next_bask_pred_model import NextBasketPredModel
from src.models.custom_layers import Attention


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
            output_dim=self._embedding_size,
            mask_zero=True,
            name=self._item_embeddings_layer_name,
        )(sequence_input)

        embedded_basket = tf.reduce_mean(embedded_sequence, axis=2)
        embedded_basket = SpatialDropout1D(0.2)(embedded_basket)

        lstm = Bidirectional(
            LSTM(
                self._lstm_units,
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=True,
            )
        )(embedded_basket)

        attention = Attention()(lstm)

        sigmoid = Dense(self._num_prods + 1, activation="sigmoid")(attention)

        self._model = Model(inputs=[sequence_input], outputs=sigmoid)
