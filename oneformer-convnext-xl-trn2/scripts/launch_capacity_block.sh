#!/usr/bin/env bash
set -euo pipefail

PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
CAPACITY_RESERVATION_ID="${CAPACITY_RESERVATION_ID:-}"
IMAGE_ID="${IMAGE_ID:-}"
SUBNET_ID="${SUBNET_ID:-}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-}"
INSTANCE_PROFILE="${INSTANCE_PROFILE:-}"
KEY_NAME="${KEY_NAME:-}"
INSTANCE_TYPE="${INSTANCE_TYPE:-trn2.3xlarge}"
INSTANCE_NAME="${INSTANCE_NAME:-oneformer-neuron-port}"
OWNER_TAG="${OWNER_TAG:-${USER:-unknown}}"
CLIENT_TOKEN="${CLIENT_TOKEN:-}"

: "${REGION:?Set AWS_REGION or AWS_DEFAULT_REGION}"
: "${CAPACITY_RESERVATION_ID:?Set CAPACITY_RESERVATION_ID}"
: "${IMAGE_ID:?Set IMAGE_ID}"
: "${SUBNET_ID:?Set SUBNET_ID}"
: "${SECURITY_GROUP_ID:?Set SECURITY_GROUP_ID}"
: "${INSTANCE_PROFILE:?Set INSTANCE_PROFILE}"
: "${KEY_NAME:?Set KEY_NAME}"
: "${CLIENT_TOKEN:?Set a unique, reusable CLIENT_TOKEN}"

aws ec2 run-instances \
    --profile "$PROFILE" \
    --region "$REGION" \
    --image-id "$IMAGE_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --subnet-id "$SUBNET_ID" \
    --security-group-ids "$SECURITY_GROUP_ID" \
    --key-name "$KEY_NAME" \
    --iam-instance-profile "Name=$INSTANCE_PROFILE" \
    --instance-market-options MarketType=capacity-block \
    --capacity-reservation-specification \
        "CapacityReservationTarget={CapacityReservationId=$CAPACITY_RESERVATION_ID}" \
    --associate-public-ip-address \
    --block-device-mappings \
        '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":512,"VolumeType":"gp3","Iops":3000,"Throughput":125,"DeleteOnTermination":true,"Encrypted":true}}]' \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled \
    --tag-specifications \
        "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Project,Value=oneformer-neuron-port},{Key=Owner,Value=$OWNER_TAG},{Key=CapacityBlock,Value=$CAPACITY_RESERVATION_ID}]" \
        "ResourceType=volume,Tags=[{Key=Name,Value=$INSTANCE_NAME-root},{Key=Project,Value=oneformer-neuron-port},{Key=Owner,Value=$OWNER_TAG}]" \
    --client-token "$CLIENT_TOKEN" \
    --count 1 \
    --query 'Instances[0].{InstanceId:InstanceId,State:State.Name,AZ:Placement.AvailabilityZone,PrivateIp:PrivateIpAddress}' \
    --output json
