import sys

sys.path.insert(1, "/home/ubuntu/sequence_models")

import os
from pyspark.sql import SparkSession
from src.data_preprocessing.data_preprocessing import *
from utils.general import *
from utils.logging_framework import log
from utils.general import training_data_to_s3

if __name__ == "__main__":

    task = sys.argv[1]
    bucket = sys.argv[2]
    staging_path = sys.argv[3]
    sample = sys.argv[4]
    sample_rate = float(sys.argv[5])
    num_prods = int(sys.argv[6])
    aws_creds = sys.argv[7]

    log.info("Running task {}".format(task))

    spark = SparkSession.builder.appName("sequence-models").getOrCreate()

    # ========== Import transaction data from staging ==========
    staging_trans_path = os.path.join(staging_path, "trans-data/")
    log.info("Importing transaction staging data from")
    trans_df = spark.read.parquet(staging_trans_path)

    # ========== Sample DataFrame if requested ==========

    if sample == "True":
        log.info(
            "Sampling transaction DataFrame with sample rate {}".format(sample_rate)
        )

        trans_df = sample_custs(trans_df, sample_rate)

    # ========== Create the sequence lists needed for the algorithm ==========
    log.info("Generating customer/basket/item lists for algorithm")
    trans_df = trans_df.orderBy("CUST_CODE")
    cust_list, item_list, customer_id_list = create_item_cust_arrays(trans_df)

    log.info("First record in cust_list")
    print(cust_list[0])

    log.info("First record in item_list")
    print(item_list[0])

    log.info("First record in customer_id_list")
    print(customer_id_list[0])

    # ========== Create dictionaries and convert PROD_CODE to indices ==========
    log.info("Creating dictionaries and converting PROD_CODE to indices")
    prod_dictionary, reversed_prod_dictionary = generate_prod_dictionaries(item_list, num_prods)

    # ========== Upload training data to s3 ==========
    log.info("Uploading training data, product and customer index mappings to s3")

    training_data_to_s3(obj=cust_list, bucket=bucket, key="cust_list.txt")
    training_data_to_s3(obj=item_list, bucket=bucket, key="item_list.txt")
    training_data_to_s3(obj=customer_id_list, bucket=bucket, key="cust_id.txt")
    training_data_to_s3(obj=prod_dictionary, bucket=bucket, key="prod_dictionary.pkl")
    training_data_to_s3(
        obj=reversed_prod_dictionary, bucket=bucket, key="reversed_prod_dictionary.pkl"
    )
