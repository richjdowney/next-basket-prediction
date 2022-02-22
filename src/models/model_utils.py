from utils.logging_framework import log
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
import numpy as np
import random


def evaluate(model: Model, test_data: np.array, eval_samp_rate: float):
    """
    Function to generate accuracy, precision, recall and f1 scores from a TensorFlow model

    Parameters
    ----------
    model: tensorflow.keras.models.Model
      Trained TensorFlow model
    test_data: np.array
      array containing lists of x and y values
    eval_samp_rate: float
      sample the evaluation sets - no sampling if set to 0 - necessary due to memory limits
      for large evaluation data sets

    """

    log.info("Evaluating model on test data")

    x = test_data[0]
    y = test_data[1]

    if eval_samp_rate > 0:
        temp = list(zip(x, y))
        random.shuffle(temp)
        x, y = zip(*temp)
        samp_num = round(len(x) * eval_samp_rate)
        x = x[0:samp_num]
        y = y[0:samp_num]

        log.info(f"Sampled evaluation data set to {samp_num} records")

    score = model.evaluate(x, y, verbose=0)
    log.info(f"Test loss: {score[0]} / Test accuracy: {score[1]}")

    validation_preds = model.predict(x)
    validation_preds = np.where(validation_preds > 0.5, 1, 0)

    precision = precision_score(y, validation_preds, average="macro")
    recall = recall_score(y, validation_preds, average="macro")
    f1 = f1_score(y, validation_preds, average="macro")

    log.info(f"Test precision: {precision} / Test recall: {recall} / Test F1: {f1}")