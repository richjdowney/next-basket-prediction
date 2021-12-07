import pytest
from pyspark.sql import SparkSession
import os


@pytest.fixture(scope="session")
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
