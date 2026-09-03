#!/usr/bin/env bash
set -uo pipefail

PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
CAPACITY_RESERVATION_ID="${CAPACITY_RESERVATION_ID:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"

: "${REGION:?Set AWS_REGION or AWS_DEFAULT_REGION}"
: "${CAPACITY_RESERVATION_ID:?Set CAPACITY_RESERVATION_ID}"

while true; do
    if state="$(
        aws ec2 describe-capacity-reservations \
            --profile "$PROFILE" \
            --region "$REGION" \
            --capacity-reservation-ids "$CAPACITY_RESERVATION_ID" \
            --query 'CapacityReservations[0].State' \
            --output text 2>&1
    )"; then
        now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "$now capacity_state=$state"
        if [[ "$state" == "active" ]]; then
            set -e
            bash scripts/launch_capacity_block.sh
            bash scripts/orchestrate_after_launch.sh
            break
        fi
    else
        now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "$now capacity_query_retry=$state"
    fi
    sleep "$POLL_SECONDS"
done
