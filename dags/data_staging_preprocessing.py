# Add path to modules to sys path
import sys

sys.path.insert(1, "/home/ubuntu/sequence_models")

from airflow import DAG
from airflow.providers.amazon.aws.operators.emr_create_job_flow import (
    EmrCreateJobFlowOperator,
)
from airflow.contrib.sensors.emr_step_sensor import EmrStepSensor
from airflow.contrib.operators.emr_add_steps_operator import EmrAddStepsOperator
from airflow.contrib.operators.emr_terminate_job_flow_operator import (
    EmrTerminateJobFlowOperator,
)
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from config.load_config import load_yaml
from config import constants
from config.load_config import Config
from utils.send_email import notify_email
from utils.logging_framework import log
from utils.copy_app_to_s3 import copy_app_to_s3

# Load the config file
config = load_yaml(constants.config_path)

# Check the config types
try:
    Config(**config)
except TypeError as error:
    log.error(error)

with DAG(**config["pre_processing_dag"]) as dag:

    # Create egg file
    create_egg = BashOperator(
        task_id="create_app_egg",
        bash_command="cd /home/ubuntu/sequence_models && python /home/ubuntu/sequence_models/setup.py bdist_egg",
        run_as_user="airflow",
    )

    # Copy application files to s3
    upload_code = PythonOperator(
        task_id="upload_app_to_s3", python_callable=copy_app_to_s3, op_args=[config]
    )

    # Start the cluster for data prep
    data_prep_cluster_creator = EmrCreateJobFlowOperator(
        task_id="data_model_preprocessing_job_flow",
        job_flow_overrides=config["emr"],
        aws_conn_id="aws_default",
        emr_conn_id="emr_default",
        on_failure_callback=notify_email,
    )

    # ========== DATA STAGING ==========
    task = "data_staging"
    data_staging = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='data_model_preprocessing_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run data staging step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["StageRunner"],
                        task,
                        config["s3"]["Bucket"],
                        config["s3"]["DataFolder"],
                        config["s3"]["StagingDataPath"],
                        "{{ execution_date }}",
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    data_staging_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('data_model_preprocessing_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # ========== DATA PRE-PROCESSING ==========
    task = "data_preprocessing"
    data_preprocessing = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='data_model_preprocessing_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run data pre-processing step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["DataPreProcessingRunner"],
                        task,
                        config["s3"]["Bucket"],
                        config["s3"]["StagingDataPath"],
                        config["datapreprocessing"]["sample"],
                        config["datapreprocessing"]["sample_rate"],
                        config["datapreprocessing"]["num_prods"],
                        config["airflow"]["AwsCredentials"],
                        "{{ execution_date }}",
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    data_preprocessing_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('data_model_preprocessing_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # ========== MODEL PRE-PROCESSING ==========
    task = "model_preprocessing"
    model_preprocessing = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='data_model_preprocessing_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run model pre-processing step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["ModelPreProcessingRunner"],
                        task,
                        config["s3"]["Bucket"],
                        config["lstmmodel"]["max_seq_length"],
                        config["lstmmodel"]["max_items_in_bask"],
                        config["datapreprocessing"]["num_prods"],
                        config["airflow"]["AwsCredentials"],
                        "{{ execution_date }}",
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    model_preprocessing_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('data_model_preprocessing_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # Remove the EMR cluster
    cluster_remover = EmrTerminateJobFlowOperator(
        task_id="remove_EMR_cluster",
        job_flow_id="{{ task_instance.xcom_pull(task_ids='data_model_preprocessing_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    create_egg >> upload_code >> data_prep_cluster_creator >> data_staging >> data_staging_step_sensor >> \
        data_preprocessing >> data_preprocessing_step_sensor >> model_preprocessing >> model_preprocessing_step_sensor \
    >> cluster_remover


