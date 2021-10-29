import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

max_seq_length = 3
max_items_in_bask = 5
num_prods = 50
embedding_size = 20
lstm_units = 10

x = np.array([[[3, 20, 23, 0, 0], [5, 42, 32, 0, 10], [24, 0, 0, 0, 0]]])

# Convert the y labels to multi-binary
y = [[0, 23, 20, 0, 0]]
y_tuple = tuple(y)
mlb = MultiLabelBinarizer(classes=[*range(0, num_prods+1, 1)])
y_multi_label = mlb.fit_transform(y_tuple)

sequence_input = Input(shape=(max_seq_length,
                              max_items_in_bask,
                              )
                       )

embedded_sequence = Embedding(
                input_dim=num_prods+1,
                input_shape=(max_items_in_bask,),
                input_length=max_seq_length,
                output_dim=embedding_size,
                mask_zero=True,
                name="item_embedding",
            )(sequence_input)

embedded_basket = tf.reduce_mean(embedded_sequence, axis=1)

lstm = LSTM(lstm_units, return_sequences=False)(embedded_basket)

softmax = Dense(num_prods + 1, activation="softmax")(lstm)

model = Model(inputs=[sequence_input], outputs=softmax)

optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"],
            )

model.fit(x, y_multi_label)