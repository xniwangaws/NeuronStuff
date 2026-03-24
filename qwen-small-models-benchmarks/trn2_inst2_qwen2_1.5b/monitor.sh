#!/bin/bash
# Monitor benchmark progress, update summary every 3 minutes
BASEDIR="/home/ubuntu/test-bytedance"
PROGRESS="$BASEDIR/benchmark_progress.txt"
SUMMARY="$BASEDIR/benchmark_summary.txt"

update() {
  {
    echo "# Qwen2-1.5B Benchmark Summary"
    echo "# Updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    printf "%-40s %-14s %-10s %-10s %-10s %-8s %-10s %-12s\n" \
      "Task" "Status" "TTFT(ms)" "TPOT(ms)" "ITL(ms)" "Req/s" "OutTok/s" "TotalTok/s"
    printf "%s\n" "---------------------------------------------------------------------------------------------------------------------------"

    for dir in "$BASEDIR"/qwen2_1.5b_*/; do
      task=$(basename "$dir")
      bench="$dir/bench.log"

      if grep -q "^DONE $task" "$PROGRESS" 2>/dev/null; then
        status="DONE"
      elif grep -q "$task" "$PROGRESS" 2>/dev/null; then
        status=$(grep "$task" "$PROGRESS" | head -1 | awk '{print $1}')
      elif [ -s "$dir/serve.log" ]; then
        if grep -q "Uvicorn running" "$dir/serve.log" 2>/dev/null; then
          status="SERVING"
        elif grep -q "Compilation Successfully" "$dir/serve.log" 2>/dev/null; then
          status="LOADING"
        elif grep -q "Compiling" "$dir/serve.log" 2>/dev/null; then
          status="COMPILING"
        else
          status="STARTING"
        fi
      else
        status="PENDING"
      fi

      if [[ "$status" == "DONE" ]] && [[ -f "$bench" ]]; then
        ttft=$(grep "Mean TTFT" "$bench" | awk '{print $NF}')
        tpot=$(grep "Mean TPOT" "$bench" | awk '{print $NF}')
        itl=$(grep "Mean ITL" "$bench" | awk '{print $NF}')
        reqs=$(grep "Request throughput" "$bench" | awk '{print $NF}')
        toks=$(grep "Output token throughput" "$bench" | head -1 | awk '{print $NF}')
        total_toks=$(grep "Total token throughput" "$bench" | awk '{print $NF}')
        printf "%-40s %-14s %-10s %-10s %-10s %-8s %-10s %-12s\n" \
          "$task" "$status" "$ttft" "$tpot" "$itl" "$reqs" "$toks" "$total_toks"
      else
        printf "%-40s %-14s\n" "$task" "$status"
      fi
    done

    echo ""
    done_count=$(grep -c '^DONE' "$PROGRESS" 2>/dev/null || echo 0)
    fail_count=$(grep -c '^FAILED' "$PROGRESS" 2>/dev/null || echo 0)
    echo "# Done: $done_count | Failed: $fail_count | Total: 24"
  } > "$SUMMARY"
}

while true; do
  update
  echo "[$(date '+%H:%M:%S')] Summary updated. Done: $(grep -c '^DONE' "$PROGRESS" 2>/dev/null || echo 0)/24"
  cat "$SUMMARY"
  echo ""

  # Stop if orchestrator is done
  if ! pgrep -f "run_all_benchmarks" > /dev/null 2>&1; then
    echo "Orchestrator finished."
    update
    cat "$SUMMARY"
    break
  fi

  sleep 180
done
