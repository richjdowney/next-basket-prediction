from utils.aws_utils import load_file_to_s3
from utils.logging_framework import log


def copy_app_to_s3(*op_args) -> log:
    """Runner to copy application files to s3
    Parameters
    ----------
    op_args : dict
        config dictionary
    """
    config = op_args[0]

    # Upload egg file to s3
    load_file_to_s3(
        file_name="{}{}".format(config["app"]["PathToEgg"], config["app"]["EggObject"]),
        bucket=config["s3"]["Bucket"],
        aws_credentials_id=config["airflow"]["AwsCredentials"],
        object_name="application/{}".format(config["app"]["EggObject"]),
    )

    # Upload runner for data staging to s3
    load_file_to_s3(
        file_name="{}{}".format(config["app"]["PathToRunners"], config["app"]["StageRunner"]),
        bucket=config["s3"]["Bucket"],
        aws_credentials_id=config["airflow"]["AwsCredentials"],
        object_name="application/{}".format(config["app"]["StageRunner"]),
    )

    # Upload runner for data pre-processing to s3
    load_file_to_s3(
        file_name="{}{}".format(config["app"]["PathToRunners"], config["app"]["DataPreProcessingRunner"]),
        bucket=config["s3"]["Bucket"],
        aws_credentials_id=config["airflow"]["AwsCredentials"],
        object_name="application/{}".format(config["app"]["DataPreProcessingRunner"]),
    )

    # Upload runner for model pre-processing to s3
    load_file_to_s3(
        file_name="{}{}".format(config["app"]["PathToRunners"], config["app"]["ModelPreProcessingRunner"]),
        bucket=config["s3"]["Bucket"],
        aws_credentials_id=config["airflow"]["AwsCredentials"],
        object_name="application/{}".format(config["app"]["ModelPreProcessingRunner"]),
    )

    # Upload runner for LSTM model training to s3
    load_file_to_s3(
        file_name="{}{}".format(config["app"]["PathToRunners"], config["app"]["LSTMModelTrainRunner"]),
        bucket=config["s3"]["Bucket"],
        aws_credentials_id=config["airflow"]["AwsCredentials"],
        object_name="application/{}".format(config["app"]["LSTMModelTrainRunner"]),
    )

    # Upload requirements
    load_file_to_s3(
        file_name="{}{}".format(config["app"]["RootPath"], config["app"]["Requirements"]),
        bucket=config["s3"]["Bucket"],
        aws_credentials_id=config["airflow"]["AwsCredentials"],
        object_name="bootstrap/{}".format(config["app"]["Requirements"]),
    )

    # Upload bootstrap shell script for dependencies
    load_file_to_s3(
        file_name="{}{}".format(config["app"]["PathToBin"], config["app"]["DependenciesShell"]),
        bucket=config["s3"]["Bucket"],
        aws_credentials_id=config["airflow"]["AwsCredentials"],
        object_name="bootstrap/{}".format(config["app"]["DependenciesShell"]),
    )

    return log.info("Uploaded application to s3 successfully!")