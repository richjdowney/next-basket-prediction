import sys

sys.path.insert(1, "/home/ubuntu/sequence_models")

import os
import numpy as np
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook
from keras.utils.vis_utils import plot_model
from src.generators.lstm_generator import lstm_data_generator
from src.models.sequence_models import LSTMModel
from utils.logging_framework import log
from utils.general import download_s3
import boto3


def task_lstm_model_fit(
    bucket: str,
    max_seq_length: str,
    max_items_in_bask: str,
    embedding_size: int,
    lstm_units: int,
    item_embeddings_layer_name: str,
    batch_size: int,
    num_epochs: int,
    steps_per_epoch: int,
    save_path: str,
    save_item_embeddings_path: str,
    save_item_embeddings_period: int,
    early_stopping_patience: int,
    save_period: int,
    eval_samp_rate: int,
):

    # these parameters are read in originally as strings as they are passed to an EMR cluster in another
    # module and they are only accepted as step parameters if they are string - converting back to int
    max_seq_length = int(max_seq_length)
    max_items_in_bask = int(max_items_in_bask)

    aws_hook = AwsBaseHook("aws_default", client_type="s3")
    credentials = aws_hook.get_credentials()

    s3 = boto3.client(
        service_name="s3",
        aws_access_key_id=credentials.access_key,
        aws_secret_access_key=credentials.secret_key,
    )

    log.info("Download training, validation and test data from s3")

    # Download training data
    cust_list_train_x = download_s3("cust_list_train_x.txt", bucket, s3)
    cust_list_train_y = download_s3("cust_list_train_y.txt", bucket, s3)

    # Download validation data
    cust_list_valid_x = download_s3("cust_list_valid_x.txt", bucket, s3)
    cust_list_valid_y = download_s3("cust_list_valid_y.txt", bucket, s3)

    log.info("shape of valid_x {}, shape of valid_y {}".format(np.shape(cust_list_valid_x), np.shape(cust_list_valid_y)))

    # Download test data
    cust_list_test_x = download_s3("cust_list_test_x.txt", bucket, s3)
    cust_list_test_y = download_s3("cust_list_test_y.txt", bucket, s3)

    log.info("Download product dictionary mapping", bucket, s3)
    prod_dictionary = download_s3("prod_dictionary.pkl", bucket, s3)

    num_prods = len(prod_dictionary)
    log.info("number of prods is {}".format(num_prods))

    data_generator = lstm_data_generator(
        batch_size=batch_size,
        cust_list_x=cust_list_train_x,
        cust_list_y=cust_list_train_y,
        shuffle=True,
    )

    m = LSTMModel(
        num_prods=num_prods,
        max_seq_length=max_seq_length,
        max_items_in_bask=max_items_in_bask,
        embedding_size=embedding_size,
        lstm_units=lstm_units,
        item_embeddings_layer_name=item_embeddings_layer_name,
    )

    m.build()
    m.compile()

    # Plot model
    plot_model(
        m._model,
        to_file="/lstm_model_plot.png",
        show_shapes=True,
        show_layer_names=True,
    )

    # ========== Train model ==========

    m.train(
        data_generator,
        validation_data=(cust_list_valid_x, cust_list_valid_y),
        test_data=(cust_list_test_x, cust_list_test_y),
        eval_samp_rate=eval_samp_rate,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        early_stopping_patience=early_stopping_patience,
        save_path=save_path,
        save_period=save_period,
        save_item_embeddings_path=save_item_embeddings_path,
        save_item_embeddings_period=save_item_embeddings_period,
        item_embeddings_layer_name=item_embeddings_layer_name,
    )

    if save_path:
        full_path = os.path.join(save_path, "final")
        m.save(full_path)

    if save_item_embeddings_path:
        m.save_item_embeddings(save_item_embeddings_path, epoch="final")

        log.info("Uploading embeddings to s3")
        filename = save_item_embeddings_path.format("final")
        full_path = os.path.join(filename, "item_embeddings.hdf5")
        s3.upload_file(Filename=full_path, Key=full_path, Bucket=bucket)
