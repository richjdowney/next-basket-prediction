from pyspark.sql import SparkSession
import boto3
import logging
from pyspark.sql.types import (
    StringType,
    IntegerType,
    StructField,
    StructType,
    FloatType,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def iterate_bucket_items(bucket: str):
    """Generator that iterates over all objects in a given s3 bucket
    See http://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.list_objects_v2
    for return data format

    Parameters
    ----------
    bucket : str
        name of s3 bucket
    """

    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket)

    for page in page_iterator:
        if page["KeyCount"] > 0:
            for item in page["Contents"]:
                yield item


def import_data(spark: SparkSession, bucket: str, data_folder: str) -> tuple:
    """Function to import all data csv's from the S3 bucket

    Parameters
    ----------
    spark : SparkSession
        Active Spark session
    bucket : str
        name of s3 bucket
    data_folder : str
        name of the folder in the bucket containing the data

    Returns
    -------
    time_df : pyspark.sql.DataFrame
       DataFrame containing calendar lookup
    all_trans : pyspark.sql.DataFrame
        DataFrame containing customer transactions for all weeks
    """

    trans_field = [
        StructField("SHOP_WEEK", IntegerType(), True),
        StructField("SHOP_DATE", IntegerType(), True),
        StructField("SHOP_WEEKDAY", IntegerType(), True),
        StructField("SHOP_HOUR", IntegerType(), True),
        StructField("QUANTITY", IntegerType(), True),
        StructField("SPEND", FloatType(), True),
        StructField("PROD_CODE", StringType(), True),
        StructField("PROD_CODE_10", StringType(), True),
        StructField("PROD_CODE_20", StringType(), True),
        StructField("PROD_CODE_30", StringType(), True),
        StructField("PROD_CODE_40", StringType(), True),
        StructField("CUST_CODE", StringType(), True),
        StructField("CUST_PRICE_SENSITIVITY", StringType(), True),
        StructField("CUST_LIFESTAGE", StringType(), True),
        StructField("BASKET_ID", StringType(), True),
        StructField("BASKET_SIZE", StringType(), True),
        StructField("BASKET_PRICE_SENSITIVITY", StringType(), True),
        StructField("BASKET_TYPE", StringType(), True),
        StructField("BASKET_DOMINANT_MISSION", StringType(), True),
        StructField("STORE_CODE", StringType(), True),
        StructField("STORE_FORMAT", StringType(), True),
        StructField("STORE_REGION", StringType(), True),
    ]

    time_field = [
        StructField("SHOP_WEEK", IntegerType(), True),
        StructField("DATE_FROM", IntegerType(), True),
        StructField("DATE_TO", IntegerType(), True),
    ]

    trans_schema = StructType(trans_field)
    trans_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=trans_schema)

    time_schema = StructType(time_field)
    time_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=time_schema)

    # Fill DataFrame
    for i in iterate_bucket_items(bucket=bucket):
        if data_folder in i["Key"]:

            data_location = "s3a://{}/{}".format(bucket, i["Key"])

            if i["Key"] == "{}/time.csv".format(data_folder):
                logger.info("Importing data for {}".format(i["Key"]))
                time_df = spark.read.csv(data_location, header=True, schema=time_schema)

            elif "transactions" in i["Key"]:
                logger.info("Importing data for {}".format(i["Key"]))

                trans_data_wk = spark.read.csv(
                    data_location, header=True, schema=trans_schema
                )
                trans_df = trans_df.union(trans_data_wk)

    return time_df, trans_df
