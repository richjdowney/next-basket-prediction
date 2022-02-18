import yaml
from typing import Any, Dict
from utils.logging_framework import log
import pydantic


class ConfigDefaultArgs(pydantic.BaseModel):
    """Configuration for the default args when setting up the DAG"""

    owner: str
    start_date: str
    end_date: str
    depends_on_past: bool
    retries: int
    catchup: bool
    email: str
    email_on_failure: bool
    email_on_retry: bool


class ConfigModelTrainDag(pydantic.BaseModel):
    """Configuration for the DAG runs"""

    # Name for the DAG run
    dag_id: str

    # Default args for DAG run e.g. owner, start_date, end_date
    default_args: ConfigDefaultArgs

    # DAG schedule interval
    schedule_interval: str


class ConfigEmr(pydantic.BaseModel):
    """Configuration for EMR clusters"""

    Instances: Dict[str, Any]

    # EMR ec2 role
    JobFlowRole: str

    # EMR role
    ServiceRole: str

    # Cluster name
    Name: str

    # Path to save logs
    LogUri: str

    # EMR version
    ReleaseLabel: str

    # Cluster configurations
    Configurations: Dict[str, Any]

    # Path to dependencies shell script on s3
    BootstrapActions: Dict[str, Any]

    # Number of steps EMR can run concurrently
    StepConcurrencyLevel: int


class ConfigApp(pydantic.BaseModel):
    """Configuration for application paths"""

    # Path to the root directory on EC2
    RootPath: str

    # Path to the runner files
    PathToRunners: str

    # Path to the bin directory on EC2
    PathToBin: str

    # Path to the egg file on EC2
    PathToEgg: str

    # Path to the utils directory on EC2
    PathToUtils: str

    # Name of the main application egg object
    EggObject: str

    # Name of Spark runner to stage tables
    StageRunner: str

    # Name of Spark runner to pre-process data
    DataPreProcessingRunner: str

    # Name of the Spark runner to process the data for model training
    ModelPreProcessingRunner: str

    # Name of the Spark runner to train the LSTM model
    LSTMModelTrainRunner: str

    # Name of the shell script for bootstrapping
    DependenciesShell: str

    # Name of the package requirements
    Requirements: str


class ConfigDataPreProcessing(pydantic.BaseModel):
    """Configuration for data pre-processing"""

    sample: str
    sample_rate: str
    num_prods: str


class ConfigLSTMModel(pydantic.BaseModel):
    """Configuration for LSTM model parameters"""

    max_seq_length: str
    max_items_in_bask: str
    embedding_size: int
    lstm_units: int
    item_embeddings_layer_name: str
    batch_size: int
    num_epochs: int
    steps_per_epoch: int
    save_path: str
    save_item_embeddings_path: str
    save_item_embeddings_period: int
    early_stopping_patience: int
    save_period: int


class ConfigAirflow(pydantic.BaseModel):
    """Configuration for Airflow access to AWS"""

    # Config for airflow defaults
    AwsCredentials: str


class ConfigS3(pydantic.BaseModel):
    """Configuration for application paths"""

    # Bucket with input data on s3
    Bucket: str

    # Folder where the input data is located
    DataFolder: str

    # Path to staging data
    StagingDataPath: str

    # Path to egg file
    egg: str

    # Path to staging tables runner file
    StageRunner: str

    # Path to pre-process data runner
    DataPreProcessingRunner: str

    # Path to runner for processing the data for model training
    ModelPreProcessingRunner: str

    # Path to runner to train the LSTM model
    LSTMModelTrainRunner: str


class Config(pydantic.BaseModel):
    """Main configuration"""

    model_train_dag: ConfigModelTrainDag
    emr: ConfigEmr
    app: ConfigApp
    datapreprocessing: ConfigDataPreProcessing
    lstmmodel: ConfigLSTMModel
    s3: ConfigS3
    airflow: ConfigAirflow


class ConfigException(Exception):
    pass


def load_yaml(config_path):

    """Function to load yaml file from path
    Parameters
    ----------
    config_path : str
        string containing path to yaml
    Returns
    ----------
    config : dict
        dictionary containing config
    """
    log.info("Importing config file from {}".format(config_path))

    if config_path is not None:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)

        log.info("Successfully imported the config file from {}".format(config_path))

    if config_path is None:
        raise ConfigException("Must supply path to the config file")

    return config
