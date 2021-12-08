import sys

sys.path.append("/home/hadoop/sequence_models")
import pytest
from pyspark.sql.types import (
    StringType,
    IntegerType,
    StructField,
    StructType,
    FloatType,
)
from src.data_staging.data_staging import import_data


@pytest.fixture(scope="function")
def get_raw_test_data(spark):
    """Imports the data for testing"""
    bucket = "sequence-models"
    data_folder = "dh-data"

    time_df, trans_df = import_data(spark, bucket, data_folder)

    return time_df, trans_df


def test_import_data(get_raw_test_data):
    """Test the import_data function
        GIVEN:  S3 bucket and key
        THEN:  Return DataFrame for time lookup with > 0 rows, 3 columns and expected schema
               Return DataFrame for transactions with > 0 rows, 22 columns and expected schema
    """

    # ======= Test DataFrames have > 0 rows =========
    time_df = get_raw_test_data[0]
    trans_df = get_raw_test_data[1]

    assert time_df.count() > 0, "time_df not created"
    assert trans_df.count() > 0, "trans_df not created"

    # ========= Test Dataframes have expected columns =========
    time_df = get_raw_test_data[0]
    trans_df = get_raw_test_data[1]

    time_df_n_expected = 3
    time_df_n_actual = len(time_df.columns)
    trans_df_n_expected = 22
    trans_df_n_actual = len(trans_df.columns)

    msg_time = f"expected ({time_df_n_expected}) and received ({time_df_n_actual}) column count for time_df does not match"
    msg_trans = f"expected ({trans_df_n_expected}) and received ({trans_df_n_actual}) column count for trans_df does not match"

    assert time_df_n_actual == time_df_n_expected, msg_time
    assert trans_df_n_actual == trans_df_n_expected, msg_trans

    # ========= Test DataFrame has expected schema =========
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

    expected_trans_schema = StructType(trans_field)

    time_field = [
        StructField("SHOP_WEEK", IntegerType(), True),
        StructField("DATE_FROM", IntegerType(), True),
        StructField("DATE_TO", IntegerType(), True),
    ]

    expected_time_schema = StructType(time_field)

    time_df = get_raw_test_data[0]
    trans_df = get_raw_test_data[1]

    assert time_df.schema == expected_time_schema, "time_df schema does not match expected schema"
    assert trans_df.schema == expected_trans_schema, "trans_df schema does not match expected schema"

