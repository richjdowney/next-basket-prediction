from utils.logging_framework import log
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np


def evaluate(model: Model, test_data: np.array):
    """
    Function to generate accuracy, precision, recall and f1 scores from a TensorFlow model

    Parameters
    ----------
    model: tensorflow.keras.models.Model
      Trained TensorFlow model
    test_data: np.array
      array containing lists of x and y values

    """

    log.info("Evaluating model on test data")

    x = test_data[0]
    y = test_data[1]

    score = model.evaluate(x, y, verbose=0)
    log.info(f"Test loss: {score[0]} / Test accuracy: {score[1]}")

    validation_preds = model.predict(x)
    validation_preds = np.where(validation_preds > 0.5, 1, 0)

    precision = precision_score(y, validation_preds, average="macro")
    recall = recall_score(y, validation_preds, average="macro")
    f1 = f1_score(y, validation_preds, average="macro")

    log.info(f"Test precision: {precision} / Test recall: {recall} / Test F1: {f1}")


def get_class_weights(y: np.array) -> tuple([np.array, np.array]):
    """Generates class weights required to balance the sample of a multi-label classification problem

    Parameters
    ----------
    y: np.array
      Array containing the "true" y values

    Returns
    -------
    negative_class_weights: np.array
        Negative example class weights
    positive_class_weight: np.array
        Positive example class weights

    """

    negative_class_weights = []
    positive_class_weights = []
    class_index = np.arange(0, np.shape(y)[1])

    for class_num in class_index:
        try:
            y_class = [item[class_num] for item in y]
            cw = compute_class_weight("balanced", classes=[0, 1], y=y_class)
            negative_class_weights.append(cw[0])
            positive_class_weights.append(cw[1])
        except:
            # Error trap if there are not negative AND positive examples in a class
            # Treats both negative and positive equally
            negative_class_weights.append(1.0)
            positive_class_weights.append(1.0)

    negative_class_weights = tf.cast(negative_class_weights, tf.float32)
    positive_class_weights = tf.cast(positive_class_weights, tf.float32)

    return negative_class_weights, positive_class_weights
