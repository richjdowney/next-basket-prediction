import sys

sys.path.insert(1, "/home/ubuntu/sequence_models")

import os
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook
from keras.utils.vis_utils import plot_model
from src.generators.lstm_generator import lstm_data_generator
from src.models.sequence_models import LSTMModel
from utils.logging_framework import log
import boto3
import pickle


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

    with open("cust_list_train_x.txt", "wb") as data:
        s3.download_fileobj(bucket, "cust_list_train_x.txt", data)
    with open("cust_list_train_x.txt", "rb") as data:
        cust_list_train_x = pickle.load(data)

    with open("cust_list_train_y.txt", "wb") as data:
        s3.download_fileobj(bucket, "cust_list_train_y.txt", data)
    with open("cust_list_train_y.txt", "rb") as data:
        cust_list_train_y = pickle.load(data)

    # Product dictionary mapping
    with open("prod_dictionary.pkl", "wb") as data:
        s3.download_fileobj(bucket, "prod_dictionary.pkl", data)
    with open("prod_dictionary.pkl", "rb") as data:
        prod_dictionary = pickle.load(data)

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

    history = m.train(
        data_generator,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        early_stopping_patience=early_stopping_patience,
        save_path=save_path,
        save_period=save_period,
        save_item_embeddings_path=save_item_embeddings_path,
        save_item_embeddings_period=save_item_embeddings_period,
        item_embeddings_layer_name=item_embeddings_layer_name,
    )

    elapsed_epochs = len(history.history["loss"])

    if save_path:
        m.save(save_path.format(epoch=elapsed_epochs))

    if save_item_embeddings_path:
        m.save_item_embeddings(save_item_embeddings_path.format(epoch=elapsed_epochs))

    log.info("Uploading embeddings to s3")
    filename = os.path.basename(save_item_embeddings_path)
    s3.upload_file(Filename=save_item_embeddings_path, Key=filename, Bucket=bucket)
