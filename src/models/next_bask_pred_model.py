from abc import ABCMeta
from utils.logging_framework import log
from src.models.model_utils import evaluate
from src.models.custom_loss import weighted_bce_loss
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import BinaryAccuracy, Recall, Precision
import h5py
import os


DEFAULT_D_MODEL = 300
DEFAULT_LSTM_UNITS = 50
DEFAULT_NUM_EPOCHS = 250
DEFAULT_STEPS_PER_EPOCH = 10000
DEFAULT_VALIDATION_STEPS = 100
DEFAULT_VALIDATION_FREQ = 1
DEFAULT_NUM_HEADS = 1
DEFAULT_DFF = 256
DEFAULT_TRANSFORMER_ENCODE = False
DEFAULT_BASKET_POOL = "dense"
DEFAULT_RUN_POS_ENCODING = False


class NextBasketPredModel(object):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        num_prods,
        max_seq_length,
        max_items_in_bask,
        num_heads,
        dff,
        transformer_encode,
        basket_pool,
        run_pos_encoding,
        item_embeddings_layer_name,
        use_class_weights,
        d_model=DEFAULT_D_MODEL,
        lstm_units=DEFAULT_LSTM_UNITS,
        negative_class_weights=None,
        positive_class_weights=None,
    ):

        self._num_prods = num_prods
        self._max_seq_length = max_seq_length
        self._max_items_in_bask = max_items_in_bask
        self._d_model = d_model
        self._num_heads = num_heads
        self._dff = dff
        self._transformer_encode = transformer_encode
        self._basket_pool = basket_pool
        self._run_pos_encoding = run_pos_encoding
        self._item_embeddings_layer_name = item_embeddings_layer_name
        self._lstm_units = lstm_units
        self._use_class_weights = use_class_weights
        self._negative_class_weights = (negative_class_weights,)
        self._positive_class_weights = (positive_class_weights,)

        self._model = None

    @property
    def item_embeddings(self):
        return _item_embeddings_from_model(
            self._model, self._item_embeddings_layer_name
        )

    def build(self):
        raise NotImplementedError()

    def compile(self, optimizer=None, learning_rate=0.01):
        if not optimizer:
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

        if self._use_class_weights:
            self._model.compile(
                optimizer=optimizer,
                loss=weighted_bce_loss(
                    negative_class_weights=self._negative_class_weights,
                    positive_class_weights=self._positive_class_weights,
                ),
                metrics=[
                    BinaryAccuracy(name="binary_accuracy", threshold=0.5),
                    Recall(top_k=20),
                    Precision(top_k=20),
                ],
            )
        else:
            self._model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=[
                    BinaryAccuracy(name="binary_accuracy", threshold=0.5),
                    Recall(top_k=20),
                    Precision(top_k=20),
                ],
            )

    def train(
        self,
        train_data_generator,
        validation_data_generator,
        test_data_generator,
        steps_per_epoch=DEFAULT_STEPS_PER_EPOCH,
        epochs=DEFAULT_NUM_EPOCHS,
        validation_steps=DEFAULT_VALIDATION_STEPS,
        validation_freq=DEFAULT_VALIDATION_FREQ,
        early_stopping_patience=None,
        reduce_learning_rate=None,
        save_path=None,
        save_period=None,
        save_item_embeddings_path=None,
        save_item_embeddings_period=None,
        item_embeddings_layer_name=None,
    ):

        callbacks = []
        if early_stopping_patience:
            callbacks.append(
                EarlyStopping(monitor="loss", patience=early_stopping_patience)
            )
        if reduce_learning_rate:
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.6,
                patience=1,
                min_lr=1e-5,
                verbose=1,
                mode="min",
            )
        if save_path and save_period:
            callbacks.append(ModelCheckpoint(save_path, save_freq=save_period))
        if save_item_embeddings_path and save_item_embeddings_period:
            callbacks.append(
                _SaveItemEmbeddings(
                    period=save_item_embeddings_period,
                    path=save_item_embeddings_path,
                    item_embeddings_layer_name=item_embeddings_layer_name,
                )
            )
        callbacks.append(_TestSetEvaluation(test_data_generator))

        history = self._model.fit(
            train_data_generator,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_data_generator,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
        )

        return history

    def save(self, path):
        log.info("Saving model to %s", path)
        self._model.save(path)

    def save_item_embeddings(self, path, epoch):
        _write_item_embeddings(self.item_embeddings, path, epoch)

    def load(self, path):
        log.info("Loading model from %s", path)
        self._model = load_model(path)


class _SaveItemEmbeddings(Callback):
    """Save item embeddings"""

    def __init__(self, period, path, item_embeddings_layer_name):
        self.period = period
        self.path = path
        self.item_embeddings_layer_name = item_embeddings_layer_name

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period != 0:
            return

        embeddings = _item_embeddings_from_model(
            self.model, self.item_embeddings_layer_name
        )
        _write_item_embeddings(embeddings, self.path, epoch)


def _item_embeddings_from_model(keras_model, item_embeddings_layer_name):
    for layer in keras_model.layers:
        if layer.get_config()["name"] == item_embeddings_layer_name:
            return layer.get_weights()[0]


def _write_item_embeddings(item_embeddings, path, epoch):
    path = path.format(epoch)
    log.info("Saving item embeddings to {}".format(path))
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, "item_embeddings.hdf5")
    with h5py.File(full_path, "w") as f:
        f.create_dataset("item_embedding_layer", data=item_embeddings)


class _TestSetEvaluation(Callback):
    """Run evaluation metrics on training end on test set"""

    def __init__(self, test_data_generator):
        self.test_data = next(test_data_generator)

    def on_train_end(self, logs={}):
        evaluate(self.model, self.test_data)
