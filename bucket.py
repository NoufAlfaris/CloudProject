import boto3

region = "us-east-1"
bucket_name = "my-async-sagemaker-results-12345"  # must be globally unique

s3_client = boto3.client("s3", region_name=region)

# Create bucket
try:
    if region == "us-east-1":
        s3_client.create_bucket(Bucket=bucket_name)
    else:
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": region}
        )
    print(f"S3 bucket '{bucket_name}' created successfully!")
except Exception as e:
    print(f"Error creating bucket: {e}")
