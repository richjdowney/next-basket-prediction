import airflow.hooks.S3_hook


def load_file_to_s3(
    file_name: str, bucket: str, aws_credentials_id: str, object_name=None
):
    """Function to upload files to s3 using Boto
    Parameters
    ----------
    file_name : str
        string containing path to file
    bucket : str
        string containing name of the s3 bucket
    aws_credentials_id : str
        name of the Airflow connection holding the AWS credentials
    object_name : str
        name of the object to upload
    """
    hook = airflow.hooks.S3_hook.S3Hook(aws_credentials_id)

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    hook.load_file(
        filename=file_name, key=object_name, bucket_name=bucket, replace=True
    )
