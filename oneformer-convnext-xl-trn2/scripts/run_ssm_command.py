#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--command", action="append", required=True)
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
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--output-file")
    return parser.parse_args()


def aws(args: argparse.Namespace, command: list[str]) -> str:
    for attempt in range(6):
        result = subprocess.run(
            [
                "aws",
                *command,
                "--profile",
                args.profile,
                "--region",
                args.region,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout

        error = (
            result.stderr.strip()
            or result.stdout.strip()
            or "AWS CLI command failed"
        )
        retryable = "Could not connect to the endpoint URL" in error
        if not retryable or attempt == 5:
            raise RuntimeError(error)
        time.sleep(min(2 ** attempt, 15))

    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
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
                        "commands": args.command,
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
    print(f"SSM command: {command_id}", flush=True)
    time.sleep(3)

    status = "Pending"
    invocation = {}
    while status in {"Pending", "InProgress", "Delayed"}:
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
        print(f"SSM status: {status}", flush=True)
        if status in {"Pending", "InProgress", "Delayed"}:
            time.sleep(10)

    stdout = invocation.get("StandardOutputContent", "")
    stderr = invocation.get("StandardErrorContent", "")
    print(stdout, end="")
    print(stderr, end="")
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(invocation, indent=2) + "\n"
        )
    if status != "Success":
        raise RuntimeError(f"SSM command failed with status {status}")


if __name__ == "__main__":
    main()
