#!/usr/bin/env python3

import argparse
import json
import os
import posixpath
import subprocess
import time

import botocore.session
from botocore.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE", "default"),
    )
    parser.add_argument(
        "--instance-region",
        default=(
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        ),
    )
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET"))
    parser.add_argument(
        "--bucket-region",
        default=os.environ.get("S3_REGION", "us-east-1"),
    )
    parser.add_argument("--expires-in", type=int, default=7200)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    if not args.bucket:
        parser.error("--bucket or S3_BUCKET is required")
    return args


def aws(args: argparse.Namespace, command: list[str]) -> str:
    result = subprocess.run(
        [
            "aws",
            *command,
            "--profile",
            args.profile,
            "--region",
            args.instance_region,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "AWS CLI command failed")
    return result.stdout


def main() -> None:
    args = parse_args()
    session = botocore.session.Session()
    session.set_config_variable("profile", args.profile)
    client = session.create_client(
        "s3",
        region_name=args.bucket_region,
        config=Config(signature_version="s3v4"),
    )
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": args.bucket, "Key": args.key},
        ExpiresIn=args.expires_in,
        HttpMethod="GET",
    )

    remote_path = args.remote_path
    part_path = f"{remote_path}.part"
    remote_dir = posixpath.dirname(remote_path)
    commands = [
        "set -e",
        f"mkdir -p {json.dumps(remote_dir)}",
        f"rm -f {json.dumps(part_path)}",
        (
            "curl --fail --silent --show-error --retry 5 --location "
            f"--output {json.dumps(part_path)} {json.dumps(url)}"
        ),
    ]
    if args.expected_sha256:
        commands.append(
            f"echo {json.dumps(args.expected_sha256 + '  ' + part_path)} | sha256sum -c -"
        )
    commands.extend(
        [
            f"mv {json.dumps(part_path)} {json.dumps(remote_path)}",
            f"sha256sum {json.dumps(remote_path)}",
            f"ls -lh {json.dumps(remote_path)}",
        ]
    )

    command_id = json.loads(
        aws(
            args,
            [
                "ssm",
                "send-command",
                "--instance-ids",
                args.instance_id,
                "--document-name",
                "AWS-RunShellScript",
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--parameters",
                json.dumps(
                    {
                        "commands": ["; ".join(commands)],
                        "executionTimeout": [str(args.timeout_seconds)],
                    }
                ),
                "--query",
                "Command.CommandId",
                "--output",
                "json",
            ],
        )
    )
    print(f"SSM download command: {command_id}", flush=True)

    status = "Pending"
    invocation: dict[str, object] = {}
    while status in {"Pending", "InProgress", "Delayed"}:
        time.sleep(5)
        invocation = json.loads(
            aws(
                args,
                [
                    "ssm",
                    "get-command-invocation",
                    "--command-id",
                    command_id,
                    "--instance-id",
                    args.instance_id,
                    "--output",
                    "json",
                ],
            )
        )
        status = str(invocation["Status"])
        print(f"SSM download status: {status}", flush=True)

    if status != "Success":
        stderr = str(invocation.get("StandardErrorContent", "")).strip()
        raise RuntimeError(stderr or f"Download failed with status {status}")
    print(str(invocation.get("StandardOutputContent", "")), end="")


if __name__ == "__main__":
    main()
