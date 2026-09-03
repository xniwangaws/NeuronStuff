#!/usr/bin/env python3

import argparse
import os

import botocore.session
from botocore.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE", "default"),
    )
    parser.add_argument(
        "--region",
        default=(
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        ),
    )
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET"))
    parser.add_argument("--key", required=True)
    parser.add_argument("--expires-in", type=int, default=3600)
    args = parser.parse_args()
    if not args.bucket:
        parser.error("--bucket or S3_BUCKET is required")
    return args


def main() -> None:
    args = parse_args()
    session = botocore.session.Session()
    session.set_config_variable("profile", args.profile)
    client = session.create_client(
        "s3",
        region_name=args.region,
        config=Config(signature_version="s3v4"),
    )
    url = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": args.bucket, "Key": args.key},
        ExpiresIn=args.expires_in,
        HttpMethod="PUT",
    )
    print(url)


if __name__ == "__main__":
    main()
