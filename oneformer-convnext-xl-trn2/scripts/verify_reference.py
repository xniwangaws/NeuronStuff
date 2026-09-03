#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actual",
        default="agent_artifacts/data/reference/metadata.json",
    )
    parser.add_argument(
        "--expected",
        default="configs/reference_fingerprint.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual = json.loads(Path(args.actual).read_text())
    expected = json.loads(Path(args.expected).read_text())
    tolerance = expected["float_tolerance"]

    for key in (
        "pixel_values_shape",
        "task_inputs_shape",
        "task_inputs_sum",
    ):
        if actual[key] != expected[key]:
            raise ValueError(
                f"Reference fingerprint mismatch for {key}: "
                f"{actual[key]} != {expected[key]}"
            )

    for key in ("pixel_values_mean", "pixel_values_std"):
        if abs(actual[key] - expected[key]) > tolerance:
            raise ValueError(
                f"Reference fingerprint mismatch for {key}: "
                f"{actual[key]} != {expected[key]}"
            )

    print(
        json.dumps(
            {
                "passed": True,
                "actual": actual,
                "expected_input_fingerprint": expected,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
