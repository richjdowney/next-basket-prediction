from utils.logging_framework import log
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
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
    score = model.evaluate(test_data[0], test_data[1], verbose=0)
    log.info(f"Test loss: {score[0]} / Test accuracy: {score[1]}")

    validation_preds = model.predict(test_data[0])
    validation_preds = np.where(validation_preds > 0.5, 1, 0)

    precision = precision_score(test_data[1], validation_preds, average="macro")
    recall = recall_score(test_data[1], validation_preds, average="macro")
    f1 = f1_score(test_data[1], validation_preds, average="macro")

    log.info(f"Test precision: {precision} / Test recall: {recall} / Test F1: {f1}")