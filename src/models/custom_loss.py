import tensorflow as tf
import keras.backend as K
import numpy as np


def weighted_bce_loss(
    negative_class_weights: np.array, positive_class_weights: np.array
) -> float:
    """Custom loss function to calculate a weighted BCE loss - required because Keras
      class_weights cannot handle multi-label input

    Parameters
    ----------
    negative_class_weights: np.array
      Array containing weights for negative classes
    positive_class_weights: np.array
      Array containing weights for positive classes

    Returns
    -------
    custom_loss: float
        Weighted BCE loss

    """

    def custom_loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # # Obtain the loss weights
        loss_weights = tf.multiply(y_true, positive_class_weights) + tf.multiply(
            (1 - y_true), negative_class_weights
        )

        bce = K.binary_crossentropy(y_true, y_pred)

        return K.mean(tf.multiply(bce, loss_weights), axis=1)

    return custom_loss
