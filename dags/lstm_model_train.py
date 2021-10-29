# Add path to modules to sys path
import sys

sys.path.insert(1, "/home/ubuntu/sequence_models")

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from config.load_config import load_yaml
from config import constants
from config.load_config import Config
from utils.send_email import notify_email
from utils.logging_framework import log
from runners import lstm_model_train_runner

# Load the config file
config = load_yaml(constants.config_path)

# Check the config types
try:
    Config(**config)
except TypeError as error:
    log.error(error)

with DAG(**config["lstm_training_dag"]) as dag:

    # ========== LSTM MODEL TRAINING ==========
    task = "lstm_model_train"
    lstm_fit = PythonOperator(
        task_id="run_lstm_model_train",
        dag=dag,
        provide_context=True,
        python_callable=lstm_model_train_runner.task_lstm_model_fit,
        op_kwargs={
            "bucket": config["s3"]["Bucket"],
            "max_seq_length": config["lstmmodel"]["max_seq_length"],
            "max_items_in_bask": config["lstmmodel"]["max_items_in_bask"],
            "embedding_size": config["lstmmodel"]["embedding_size"],
            "lstm_units": config["lstmmodel"]["lstm_units"],
            "item_embeddings_layer_name": config["lstmmodel"]["item_embeddings_layer_name"],
            "batch_size": config["lstmmodel"]["batch_size"],
            "num_epochs": config["lstmmodel"]["num_epochs"],
            "steps_per_epoch": config["lstmmodel"]["steps_per_epoch"],
            "save_path": config["lstmmodel"]["save_path"],
            "save_item_embeddings_path": config["lstmmodel"]["save_item_embeddings_path"],
            "save_item_embeddings_period": config["lstmmodel"]["save_item_embeddings_period"],
            "early_stopping_patience": config["lstmmodel"]["early_stopping_patience"],
            "save_period": config["lstmmodel"]["save_period"],
        },
        on_failure_callback=notify_email,
    )

    lstm_fit
