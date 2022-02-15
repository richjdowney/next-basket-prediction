import sys

sys.path.insert(1, "/home/ubuntu/sequence_models")

import boto3
from utils.logging_framework import log
from src.model_preprocessing.model_preprocessing import *
from utils.general import training_data_to_s3, download_s3


if __name__ == "__main__":

    task = sys.argv[1]
    bucket = sys.argv[2]
    max_seq_length = int(sys.argv[3])
    max_items_in_bask = int(sys.argv[4])
    num_prods = int(sys.argv[5])

    # ========== Download the training data from s3 ==========

    log.info("Download data and dictionaries from s3")
    s3 = boto3.client("s3")

    # List of customer / basket / item
    cust_bask_item_list = download_s3("cust_list.txt", bucket, s3)

    # Customer index list (lists of customers in same order as all_cust_data)
    cust_id_list = download_s3("cust_id.txt", bucket, s3)

    # Product dictionary mapping
    prod_dictionary = download_s3("prod_dictionary.pkl", bucket, s3)

    # Reversed product dictionary mapping
    reversed_prod_dictionary = download_s3("reversed_prod_dictionary.pkl", bucket, s3)

    log.info("First 5 records in cust_bask_item_list")
    log.info(cust_bask_item_list[0:5])

    log.info("First 5 records in cust_id_list")
    log.info(cust_id_list[0:5])

    log.info("First 5 keys in prod_dictionary")
    log.info(list(prod_dictionary.keys())[0:5])

    log.info("First 5 keys in reversed_prod_dictionary")
    log.info(list(reversed_prod_dictionary.keys())[0:5])

    # Convert PROD_CODE to integers using stored dictionaries
    log.info("Converting PROD_CODE to integers with stored dictionaries")
    cust_list_index = convert_prod_to_index(cust_bask_item_list, prod_dictionary)

    # Pad all baskets to a standard length i.e. each customer basket will have the same number of items
    # with some padded
    log.info("Padding baskets to standard length")
    cust_list_item_pad = pad_baskets(cust_list_index, max_items_in_bask)

    # Split the data into training, testing and validation sets
    log.info("Splitting data to training, testing and validation sets")
    cust_list_train, cust_list_test, cust_list_valid = create_test_valid(
        cust_list_item_pad, [0.6, 0.2, 0.2]
    )

    log.info("First record from cust_list_train")
    log.info(cust_list_train[0])

    log.info("First record from cust_list_test")
    log.info(cust_list_test[0])

    log.info("First record from cust_list_valid")
    log.info(cust_list_valid[0])

    # Create x and y datasets - y will be the final basket in the sequence as this will be a 'next basket' prediction
    log.info("Creating x and y datasets - y will be the final basket in the sequence")
    cust_list_train_x, cust_list_train_y = create_x_y_list(
        cust_list_train, num_prods, max_seq_length
    )
    cust_list_test_x, cust_list_test_y = create_x_y_list(
        cust_list_test, num_prods, max_seq_length
    )
    cust_list_valid_x, cust_list_valid_y = create_x_y_list(
        cust_list_valid, num_prods, max_seq_length
    )

    # Pad all customer x sequences to a standard length i.e. all customers will have the same number of
    # transactions (with some padded)
    log.info("Padding all customer x sequences")
    cust_list_train_x = pad_cust_seq(
        cust_list_train_x, max_seq_length, max_items_in_bask
    )
    cust_list_test_x = pad_cust_seq(cust_list_test_x, max_seq_length, max_items_in_bask)
    cust_list_valid_x = pad_cust_seq(
        cust_list_valid_x, max_seq_length, max_items_in_bask
    )

    # After processing the x and y lists must be equal
    assert len(cust_list_train_x) == len(cust_list_train_y), "Training x and y are not equal"
    assert len(cust_list_test_x) == len(cust_list_test_y), "Test x and y are not equal"
    assert len(cust_list_valid_x) == len(cust_list_valid_y), "Validation x and y are not equal"

    # Upload all training, testing and validation data
    log.info("Uploading training, testing and validation model data to s3")
    training_data_to_s3(
        obj=cust_list_train_x, bucket=bucket, key="cust_list_train_x.txt"
    )
    training_data_to_s3(
        obj=cust_list_train_y, bucket=bucket, key="cust_list_train_y.txt"
    )
    training_data_to_s3(
        obj=cust_list_test_x, bucket=bucket, key="cust_list_test_x.txt"
    )
    training_data_to_s3(
        obj=cust_list_test_y, bucket=bucket, key="cust_list_test_y.txt"
    )
    training_data_to_s3(
        obj=cust_list_valid_x, bucket=bucket, key="cust_list_valid_x.txt"
    )
    training_data_to_s3(
        obj=cust_list_valid_y, bucket=bucket, key="cust_list_valid_y.txt"
    )
