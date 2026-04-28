# SOP: Capacity-Block Survival on Trn2

Prerequisite for any multi-hour benchmark on trn2.48xlarge capacity blocks. Learned the hard way during this work (lost 45 min to silent upload failures + one preprocess round due to deleting the only local copy of the HF checkpoint before S3 sync completed).

## The problem

Trn2.48xlarge capacity blocks **force-terminate at `EndDate`**. EBS default is `DeleteOnTermination=true`, ephemeral NVMe never persists. AWS writes to `http://169.254.169.254/latest/meta-data/instance-action` ~2 minutes before SIGKILL.

**You cannot sync 300 GB in a 2-minute warning window.** Any strategy that relies on a shutdown hook is doomed. The only working approach is **continuous produce-and-sync from the moment the instance boots**.

## Three-daemon pattern

Run all three at instance setup time, before any real work begins.

### Layer A — 5-min incremental sync (catches almost everything)

```bash
AK=$(aws configure get aws_access_key_id)
SK=$(aws configure get aws_secret_access_key)
ssh -i ~/.ssh/neuron-bench-<region>.pem ubuntu@<ip> "cat > ~/.aws_env <<EOF
export AWS_ACCESS_KEY_ID=$AK
export AWS_SECRET_ACCESS_KEY=$SK
export AWS_DEFAULT_REGION=us-east-2
EOF
chmod 600 ~/.aws_env"

ssh <host> 'nohup bash -c "
source ~/.aws_env
while true; do
  aws s3 sync /opt/dlami/nvme/models/ \
    s3://<your-bucket>/<project>/ \
    --exclude \"*.cache/*\" --exclude \"*.incomplete\" \
    --region us-east-2 --only-show-errors
  aws s3 sync /opt/dlami/nvme/profiles/ \
    s3://<your-bucket>/<project>/profile/ \
    --region us-east-2 --only-show-errors
  aws s3 cp /opt/dlami/nvme/*.log \
    s3://<your-bucket>/<project>/logs/ \
    --region us-east-2 --only-show-errors --recursive 2>/dev/null
  sleep 300
done
" > /opt/dlami/nvme/continuous_sync.log 2>&1 & disown'
```

**Use `--only-show-errors`, never `--quiet`.** Quiet hides silent upload failures. We lost a 45-minute upload this way.

### Layer B — real-time sync for small high-value artifacts (inotify)

```bash
ssh <host> 'sudo apt install -y inotify-tools; nohup bash -c "
source ~/.aws_env
inotifywait -m -r -e close_write \
    --format %w%f \
    /opt/dlami/nvme/profiles/ \
    /opt/dlami/nvme/ \
  | while read f; do
      case \"\$f\" in
        *.summary.txt|*.summary.json|*.log|*.json) ;;
        *) continue ;;
      esac
      rel=\"\${f#/opt/dlami/nvme/}\"
      aws s3 cp \"\$f\" \"s3://<your-bucket>/<project>/\$rel\" \
        --region us-east-2 --only-show-errors || true
    done
" > /opt/dlami/nvme/inotify_sync.log 2>&1 & disown'
```

### Layer C — termination-action watcher

```bash
ssh <host> 'nohup bash -c "
source ~/.aws_env
TOKEN=\$(curl -sX PUT http://169.254.169.254/latest/api/token \
    -H \"X-aws-ec2-metadata-token-ttl-seconds: 21600\")
while true; do
  ACTION=\$(curl -s -H \"X-aws-ec2-metadata-token: \$TOKEN\" \
    http://169.254.169.254/latest/meta-data/instance-action 2>/dev/null)
  if [[ \"\$ACTION\" == *action* ]]; then
    echo \"[\$(date -u +%FT%TZ)] TERMINATION: \$ACTION\" \
      | tee -a /opt/dlami/nvme/termination.log
    # Rush-sync SMALL files only. DO NOT attempt 300 GB here.
    aws s3 sync /opt/dlami/nvme/profiles/ \
      s3://<your-bucket>/<project>/profile/emergency/ \
      --exclude \"*.ntff\" --region us-east-2 --only-show-errors
    aws s3 cp /opt/dlami/nvme/*.log \
      s3://<your-bucket>/<project>/emergency/ \
      --region us-east-2 --only-show-errors --recursive
    break
  fi
  sleep 30
done
" > /opt/dlami/nvme/termination_watcher.log 2>&1 & disown'
```

## Sync strategy by data type

| Data | Size | When to sync |
|---|---|---|
| HF checkpoint (post-download) | 300 GB | **One-shot** `aws s3 sync` immediately after download. Static thereafter. |
| Preprocessed Neuron-FP8 / BF16 | 300–620 GB | **One-shot** sync after preprocess finishes. Static thereafter. |
| `/var/tmp/neuron-compile-cache/` | ~300 MB | Layer A (5-min cron) |
| `save_sharded_checkpoint` weights dir | 300–500 GB | **Skip during writes** (I/O contention). Sync once `model.load()` completes. |
| `.summary.txt`, `.summary.json`, `.log` | <1 MB each | Layer B (inotify realtime) |
| `.ntff` profile traces | 50 MB – 1.5 GB | Layer A next cycle picks up. Can `rm` locally after. |
| End-of-run tgz archive | varies | Manual `aws s3 cp` before terminate |

## Pre-terminate verification

**Always verify size matches before running `ec2 terminate-instances`**:

```bash
aws s3 ls s3://<bucket>/<prefix>/ --recursive --summarize | tail -3
```

Expect `Total Size` to be within a few GB of what's on `/opt/dlami/nvme`. If mismatch, run one more explicit `aws s3 sync` while the instance is healthy. **The 2-minute termination-action window is not enough for a large sync.**

## Common failure modes

1. **Passing creds as shell vars** — captures value, but child `aws` may not inherit. Always use `~/.aws_env` + `source`.
2. **`--quiet` on sync** — hides silent failures. Use `--only-show-errors`.
3. **Deleting source before S3 sync completes** — verify by listing S3 first.
4. **Relying on EBS to persist** — default is `DeleteOnTermination=true`. Only use EBS persistence (separate attached volume with `DeleteOnTermination=false`) if you can afford ~$80/month per TB.
5. **Not mounting NVMe** — DLAMI doesn't mount the ephemeral volumes. `sudo mkfs.ext4 /dev/nvme0n1 && sudo mount /dev/nvme0n1 /opt/dlami/nvme`.

## IAM role alternative

If the instance has an IAM role with S3 write permissions (preferred), skip the `~/.aws_env` credentials. Attach via `IamInstanceProfile` in `run-instances` or via `aws ec2 associate-iam-instance-profile`.
