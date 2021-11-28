import sys

sys.path.append("/home/hadoop/sequence_models")
import pytest
from pyspark.sql import SparkSession
from src.data_staging.data_staging import import_data
import os


@pytest.fixture(scope="module")
def spark():
    """Get the spark session and download Hadoop packages needed for local S3 access"""

    os.environ[
        "PYSPARK_SUBMIT_ARGS"
    ] = "--packages com.amazonaws:aws-java-sdk-pom:1.11.538,org.apache.hadoop:hadoop-aws:3.3.1 pyspark-shell"

    spark = (
        SparkSession.builder.appName("sequence-models")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )

    return spark


@pytest.fixture(scope="module")
def get_test_data(spark):
    """Imports the data for testing"""
    bucket = "sequence-models"
    data_folder = "dh-data"

    time_df, trans_df = import_data(spark, bucket, data_folder)

    return time_df, trans_df


# Tests:
#         1.)  Import function generates 2 DataFrames with row count > 0
#         2.)  The trans_df has 22 columns
#         3.)  The time_df has 3 columns
#         4.)  The schemas match expectation
def test_dfs_created(get_test_data):
    """Test that the DataFrames are created and have >0 rows"""
    time_df = get_test_data[0]
    trans_df = get_test_data[1]

    assert time_df.count() > 0
    assert trans_df.count() > 0
