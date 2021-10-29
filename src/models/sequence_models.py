import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Input,
)
from tensorflow.keras.models import Model
from src.models.next_bask_pred_model import NextBasketPredModel


class LSTMModel(NextBasketPredModel):

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
        lstm = LSTM(self._lstm_units, return_sequences=False)(embedded_basket)
        softmax = Dense(self._num_prods + 1, activation="softmax")(lstm)

        self._model = Model(inputs=[sequence_input], outputs=softmax)