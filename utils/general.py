import boto3
import pickle
import numpy as np
from pyspark.sql import DataFrame


def sample_custs(df: DataFrame, sample_rate: float) -> DataFrame:
    """Function to sample customers for modelling

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        name of the DataFrame with customer ID to sample
    sample_rate : float
        percentage of the customers to sample
    Returns
    -------
    df : pyspark.sql.DataFrame
       sampled DataFrame
    """

    cust_samp = df.select("CUST_CODE").dropDuplicates()
    cust_samp = cust_samp.sample(withReplacement=False, fraction=sample_rate, seed=42)
    df = df.join(cust_samp, "CUST_CODE", how="inner")

    return df


def training_data_to_s3(obj: any, bucket: str, key: str):
    """Function to upload the training data to s3

    Parameters
    ----------
    obj : list, np.ndarray, dict
        object to upload, either a list, numpy array or dict
    bucket : str
        name of the s3 bucket
    key : str
        name of the file to upload
    """

    bucket = bucket
    key = key
    s3c = boto3.client("s3")

    if isinstance(obj, list):
        with open(key, "wb") as obj_pickle:
            pickle.dump(obj, obj_pickle)
        s3c.upload_file(key, bucket, key)

    if isinstance(obj, dict):
        with open(key, "wb") as obj_pickle:
            pickle.dump(obj, obj_pickle)
        s3c.upload_file(key, bucket, key)

    if isinstance(obj, np.ndarray):
        np.savetxt(key, obj, delimiter=",")
        s3c.upload_file(key, bucket, key)