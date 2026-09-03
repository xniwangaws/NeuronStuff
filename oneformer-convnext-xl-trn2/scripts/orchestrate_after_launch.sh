#!/usr/bin/env bash
set -euo pipefail

PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
INSTANCE_NAME="${INSTANCE_NAME:-oneformer-neuron-port}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/oneformer-neuron-port}"

: "${REGION:?Set AWS_REGION or AWS_DEFAULT_REGION}"

instance_id="$(
    aws ec2 describe-instances \
        --profile "$PROFILE" \
        --region "$REGION" \
        --filters \
            "Name=tag:Name,Values=$INSTANCE_NAME" \
            "Name=instance-state-name,Values=pending,running" \
        --query 'Reservations[].Instances[0].InstanceId' \
        --output text
)"

if [[ -z "$instance_id" || "$instance_id" == "None" ]]; then
    echo "No pending/running instance named $INSTANCE_NAME." >&2
    exit 75
fi

echo "INSTANCE_ID=$instance_id"
aws ec2 wait instance-running \
    --profile "$PROFILE" \
    --region "$REGION" \
    --instance-ids "$instance_id"
aws ec2 wait instance-status-ok \
    --profile "$PROFILE" \
    --region "$REGION" \
    --instance-ids "$instance_id"

while true; do
    ping_status="$(
        aws ssm describe-instance-information \
            --profile "$PROFILE" \
            --region "$REGION" \
            --filters "Key=InstanceIds,Values=$instance_id" \
            --query 'InstanceInformationList[0].PingStatus' \
            --output text
    )"
    echo "SSM=$ping_status"
    if [[ "$ping_status" == "Online" ]]; then
        break
    fi
    sleep 10
done

python scripts/upload_bundle_ssm.py \
    --instance-id "$instance_id" \
    --profile "$PROFILE" \
    --region "$REGION"

python scripts/run_ssm_command.py \
    --instance-id "$instance_id" \
    --profile "$PROFILE" \
    --region "$REGION" \
    --timeout-seconds 7200 \
    --output-file agent_artifacts/results/remote_setup_ssm.json \
    --command "cd $REMOTE_DIR && sudo -u ubuntu -H bash scripts/remote_setup.sh"

python scripts/run_ssm_command.py \
    --instance-id "$instance_id" \
    --profile "$PROFILE" \
    --region "$REGION" \
    --timeout-seconds 21600 \
    --output-file agent_artifacts/results/remote_run_raw_ssm.json \
    --command "cd $REMOTE_DIR && sudo -u ubuntu -H bash scripts/remote_run_raw.sh"
