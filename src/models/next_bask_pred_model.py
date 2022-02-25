from abc import ABCMeta
from utils.logging_framework import log
from src.models.model_utils import evaluate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import h5py
import os


DEFAULT_EMBEDDING_SIZE = 300
DEFAULT_LSTM_UNITS = 50
DEFAULT_NUM_EPOCHS = 250
DEFAULT_STEPS_PER_EPOCH = 10000


class NextBasketPredModel(object):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        num_prods,
        max_seq_length,
        max_items_in_bask,
        item_embeddings_layer_name,
        embedding_size=DEFAULT_EMBEDDING_SIZE,
        lstm_units=DEFAULT_LSTM_UNITS,
    ):

        self._num_prods = num_prods
        self._max_seq_length = max_seq_length
        self._max_items_in_bask = max_items_in_bask

        self._embedding_size = embedding_size
        self._item_embeddings_layer_name = item_embeddings_layer_name
        self._lstm_units = lstm_units

        self._model = None

    @property
    def item_embeddings(self):
        return _item_embeddings_from_model(
            self._model, self._item_embeddings_layer_name
        )

    def build(self):
        raise NotImplementedError()

    def compile(self, optimizer=None):
        if not optimizer:
            optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

        self._model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def train(
        self,
        generator,
        validation_data,
        test_data,
        steps_per_epoch=DEFAULT_STEPS_PER_EPOCH,
        epochs=DEFAULT_NUM_EPOCHS,
        early_stopping_patience=None,
        save_path=None,
        save_period=None,
        save_item_embeddings_path=None,
        save_item_embeddings_period=None,
        item_embeddings_layer_name=None,
        eval_samp_rate=None,
    ):

        callbacks = []
        if early_stopping_patience:
            callbacks.append(
                EarlyStopping(monitor="loss", patience=early_stopping_patience)
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
        callbacks.append(
            _TestSetEvaluation(test_data, eval_samp_rate)
            )

        history = self._model.fit(
            generator,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(validation_data[0], validation_data[1]),
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
    def __init__(self, test_data, eval_samp_rate):
        self.test_data = test_data
        self.eval_samp_rate = eval_samp_rate

    def on_train_end(self, logs={}):
        evaluate(self.model, self.test_data, self.eval_samp_rate)



