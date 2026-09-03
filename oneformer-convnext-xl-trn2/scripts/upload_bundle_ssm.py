#!/usr/bin/env python3

import argparse
import base64
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument(
        "--bundle",
        default="agent_artifacts/tmp/oneformer-neuron-port.tar.gz",
    )
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
    parser.add_argument(
        "--remote-dir",
        default="/home/ubuntu/oneformer-neuron-port",
    )
    return parser.parse_args()


def aws(args: argparse.Namespace, command: list[str]) -> str:
    result = subprocess.run(
        [
            "aws",
            *command,
            "--profile",
            args.profile,
            "--region",
            args.region,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def run_ssm(args: argparse.Namespace, commands: list[str]) -> None:
    response = json.loads(
        aws(
            args,
            [
                "ssm",
                "send-command",
                "--instance-ids",
                args.instance_id,
                "--document-name",
                "AWS-RunShellScript",
                "--parameters",
                json.dumps({"commands": commands}),
                "--query",
                "Command.CommandId",
                "--output",
                "json",
            ],
        )
    )
    command_id = response

    status = "Pending"
    invocation = {}
    while status in {"Pending", "InProgress", "Delayed"}:
        time.sleep(3)
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
        status = invocation["Status"]

    print(invocation.get("StandardOutputContent", ""), end="")
    print(invocation.get("StandardErrorContent", ""), end="")
    if status != "Success":
        raise RuntimeError(f"SSM command failed with status {status}")


def main() -> None:
    args = parse_args()
    bundle = Path(args.bundle).read_bytes()
    encoded = base64.b64encode(bundle).decode("ascii")
    remote_base64 = "/tmp/oneformer-neuron-port.tar.gz.b64"
    chunk_size = 12000

    run_ssm(
        args,
        [
            f"install -d -o ubuntu -g ubuntu {args.remote_dir}",
            f": > {remote_base64}",
        ],
    )
    for offset in range(0, len(encoded), chunk_size):
        chunk = encoded[offset : offset + chunk_size]
        run_ssm(
            args,
            [f"printf '%s' '{chunk}' >> {remote_base64}"],
        )

    run_ssm(
        args,
        [
            f"base64 --decode {remote_base64} > /tmp/oneformer-neuron-port.tar.gz",
            f"tar -xzf /tmp/oneformer-neuron-port.tar.gz -C {args.remote_dir}",
            f"chown -R ubuntu:ubuntu {args.remote_dir}",
            f"rm -f {remote_base64} /tmp/oneformer-neuron-port.tar.gz",
            f"find {args.remote_dir} -maxdepth 2 -type f -print | sort",
        ],
    )


if __name__ == "__main__":
    main()
