import sys
sys.path.insert(1, "/home/ubuntu/sequence_models")

import os
from src.data_staging.data_staging import *
from src.data_staging.data_profiling import DataProfiling
from utils.logging_framework import log

if __name__ == "__main__":

    task = sys.argv[1]
    bucket = sys.argv[2]
    data_folder = sys.argv[3]
    staging_path = sys.argv[4]

    spark = SparkSession.builder.appName("sequence-models").getOrCreate()

    log.info("Running task {}".format(task))

    # ========== Import the time and transaction DataFrames ==========

    log.info("Reading data files from s3 bucket {}".format(bucket))
    time_df, trans_df = import_data(spark, bucket, data_folder)

    log.info(
        "Transaction DataFrame has {} unique baskets and {} unique products over {} weeks".format(
            trans_df.select("BASKET_ID").dropDuplicates().count(),
            trans_df.select("PROD_CODE").dropDuplicates().count(),
            trans_df.select("SHOP_WEEK").dropDuplicates().count(),
        )
    )

    # ========== Profile the DataFrames ==========

    # Transaction DataFrame
    log.info("Profiling transaction DataFrame")

    data_profiler_trans = DataProfiling(df=trans_df, df_desc="trans_df")
    data_profiler_trans.top10_records()
    data_profiler_trans.print_schema()
    data_profiler_trans.row_column_counts()
    data_profiler_trans.check_df_missing()
    data_profiler_trans.check_missing_per_col()
    data_profiler_trans.top20_string_values()
    data_profiler_trans.num_col_profile()

    # Time DataFrame
    log.info("Profiling time DataFrame")

    data_profiler_time = DataProfiling(df=time_df, df_desc="time_df")
    data_profiler_time.top10_records()
    data_profiler_time.print_schema()
    data_profiler_time.row_column_counts()
    data_profiler_time.check_df_missing()
    data_profiler_time.check_missing_per_col()
    data_profiler_time.top20_string_values()
    data_profiler_time.num_col_profile()

    # ========== Write to Staging ==========
    staging_trans_path = os.path.join(staging_path, "trans-data/")
    log.info("Writing transaction data to {}".format(staging_trans_path))
    trans_df.repartition(20, "SHOP_WEEK")
    trans_df.write.parquet(staging_trans_path, mode="overwrite")

    staging_time_path = os.path.join(staging_path, "time-data/")
    log.info("Writing time data to {}".format(staging_time_path))
    time_df.repartition(20, "SHOP_WEEK")
    time_df.write.parquet(staging_time_path, mode="overwrite")