# main.tf

# Define provider
provider "aws" {
  region = "us-west-2" # Change to your desired region
}

# Create S3 bucket
resource "aws_s3_bucket" "my_bucket" {
  bucket = "reddit-yifugao" # Replace with your desired bucket name
  acl    = "private"               # Set access control list (ACL) for the bucket, e.g., private, public-read, etc.
}
