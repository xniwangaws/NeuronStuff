#!/bin/bash
# Auto-update benchmark_results.md from bench.log files
# Section-aware: uses folder name to determine which section (4K/8K/16K/32K/64K) to update
BASEDIR="/home/ubuntu/test-bytedance"
MD="$BASEDIR/benchmark_results.md"
PROGRESS="$BASEDIR/benchmark_progress.txt"

python3 << 'PYEOF'
import re, os

BASEDIR = "/home/ubuntu/test-bytedance"
MD = os.path.join(BASEDIR, "benchmark_results.md")
PROGRESS = os.path.join(BASEDIR, "benchmark_progress.txt")

# Read progress file to find DONE and FAILED tasks
done_tasks = set()
failed_tasks = {}  # folder -> reason
if os.path.exists(PROGRESS):
    with open(PROGRESS) as f:
        for line in f:
            line = line.strip()
            if line.startswith("DONE "):
                done_tasks.add(line.split(" ", 1)[1])
            elif line.startswith("FAILED"):
                parts = line.split(" ", 1)
                reason = parts[0]  # e.g. FAILED_SERVE, FAILED_TIMEOUT, FAILED_BENCH
                folder = parts[1] if len(parts) > 1 else ""
                failed_tasks[folder] = reason

# Task metadata: folder_name -> (bs, tp, lnc, section_label)
TASK_META = {}
for size_label, input_sizes in [("4K", ["4k"]), ("8K", ["8k"]), ("16K", ["16k"]), ("32K", ["32k"]), ("64K", ["64k"])]:
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        for tp in [1, 2, 4]:
            for lnc in [1, 2]:
                folder = f"qwen2_1.5b_{input_sizes[0]}_bs{bs}_tp{tp}_lnc{lnc}"
                TASK_META[folder] = (str(bs), str(tp), str(lnc), size_label)

# Extract results from bench.log files (DONE tasks)
results = {}  # (section_label, bs, tp, lnc) -> new_row_string
for task in done_tasks:
    if task not in TASK_META:
        continue
    bench = os.path.join(BASEDIR, task, "bench.log")
    if not os.path.exists(bench):
        continue

    bs, tp, lnc, label = TASK_META[task]
    with open(bench) as f:
        content = f.read()

    itl = re.search(r"Mean ITL \(ms\):\s+([\d.]+)", content)
    ttft = re.search(r"Mean TTFT \(ms\):\s+([\d.]+)", content)
    reqs = re.search(r"Request throughput \(req/s\):\s+([\d.]+)", content)
    toks = re.search(r"Output token throughput \(tok/s\):\s+([\d.]+)", content)

    if not all([itl, ttft, reqs, toks]):
        continue

    itl_val = itl.group(1)
    ttft_s = f"{float(ttft.group(1))/1000:.2f}"
    reqmin = f"{float(reqs.group(1))*60:.1f}"
    toks_val = toks.group(1)

    new_row = f"| Trn2 | - | BF16 | {bs} | {tp} | {lnc} | {itl_val} | {ttft_s} | - | {reqmin} | {toks_val} |"
    results[(label, bs, tp, lnc)] = new_row

# Add FAILED tasks
for task, reason in failed_tasks.items():
    if task not in TASK_META:
        continue
    bs, tp, lnc, label = TASK_META[task]
    key = (label, bs, tp, lnc)
    if key not in results:  # don't overwrite a DONE result
        # Map reason to readable label
        reason_map = {
            "FAILED_SERVE": "FAILED (server crash)",
            "FAILED_TIMEOUT": "FAILED (compile timeout)",
            "FAILED_BENCH": "FAILED (bench error)",
            "FAILED_OOM": "FAILED (HBM OOM, need TP≥2)",
        }
        reason_label = reason_map.get(reason, f"FAILED ({reason})")
        new_row = f"| Trn2 | - | BF16 | {bs} | {tp} | {lnc} | {reason_label} | | | | |"
        results[key] = new_row

if not results:
    print("No results to update.")
    exit(0)

# Read MD file and update section-aware
with open(MD) as f:
    lines = f.readlines()

current_section = None
updated = 0
for i, line in enumerate(lines):
    # Detect section headers like "### 4K (" or "### 64K ("
    m = re.match(r"^### (\d+K)\s", line)
    if m:
        current_section = m.group(1)
        continue

    if current_section is None:
        continue

    # Match empty rows or FAILED rows
    row_match = re.match(
        r"\| Trn2 \| - \| BF16 \| (\d+) \| (\d+) \| (\d+) \| (?:FAILED[^|]*| *)\|",
        line
    )
    if row_match:
        bs, tp, lnc = row_match.group(1), row_match.group(2), row_match.group(3)
        key = (current_section, bs, tp, lnc)
        if key in results:
            lines[i] = results[key] + "\n"
            updated += 1
            print(f"Updated: {current_section} BS={bs} TP={tp} LNC={lnc}")

with open(MD, "w") as f:
    f.writelines(lines)

print(f"\nDone. Updated {updated} rows in {MD}")
PYEOF
