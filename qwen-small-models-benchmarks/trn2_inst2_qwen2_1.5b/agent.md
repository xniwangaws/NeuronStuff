# Qwen2-1.5B Benchmark Agent Notes

## Model
- Path: `/home/ubuntu/test-bytedance/Qwen2-1.5B/`
- Framework: `neuronx-distributed-inference` via vLLM
- Dtype: BF16
- Machine: trn2.48xlarge (16 devices × 4 cores = 64 NeuronCores)

## Current Scope: TP=1 only (30 tasks)
TP=2, TP=4 configs deleted for now, will add back later.

## Configurations
| Config | tp_degree | logical_nc_config | NEURON_RT_NUM_CORES |
|--------|-----------|-------------------|---------------------|
| tp1_lnc1 | 1 | 1 | 1 |
| tp1_lnc2 | 1 | 2 | 2 |

Formula: `NEURON_RT_NUM_CORES = tp_degree × logical_nc_config`

## Input Length / Bucket Mapping
Input lengths are reduced from bucket size to avoid overflow with `--random-range-ratio 0.1` (max input = input_len × 1.1 must < context_bucket).

| Label | input_len | context_encoding_bucket | token_generation_bucket (=seq_len=max_model_len) |
|-------|-----------|------------------------|---------------------------------------------------|
| 4K | 3700 | 4096 | 4396 |
| 8K | 7400 | 8192 | 8492 |
| 16K | 14800 | 16384 | 16684 |
| 32K | 29700 | 32768 | 33068 |
| 64K | 59500 | 65536 | 65836 |

seq_len = context_encoding_bucket + 300 (OUTPUT_LEN)

## Batch Sizes per Input Length
| Input | Batch Sizes |
|-------|-------------|
| 4K | 16, 32, 64 |
| 8K | 8, 16, 32 |
| 16K | 1, 2, 4 |
| 32K | 1, 2, 4 |
| 64K | 1, 2, 4 |

## Execution: Parallel Wave Model
- **Wave 1**: 15× LNC=1 tasks in parallel (cores 0-14, ports 8080-8094)
- **Wave 2**: 15× LNC=2 tasks in parallel (cores 0-29, ports 8080-8094)
- Each task has its own `BASE_COMPILE_WORK_DIR` → `{task_dir}/compile_cache/`
- Wave 2 starts after Wave 1 fully completes (15s cooldown)
- `benchmark_progress.txt` tracks DONE/FAILED for resume support

## Neuron Core & Port Assignment

### Wave 1 (LNC=1): 15 tasks, 1 core each
| Task | Core | Port |
|------|------|------|
| 4k_bs16_lnc1 | 0 | 8080 |
| 4k_bs32_lnc1 | 1 | 8081 |
| 4k_bs64_lnc1 | 2 | 8082 |
| 8k_bs8_lnc1 | 3 | 8083 |
| 8k_bs16_lnc1 | 4 | 8084 |
| 8k_bs32_lnc1 | 5 | 8085 |
| 16k_bs1_lnc1 | 6 | 8086 |
| 16k_bs2_lnc1 | 7 | 8087 |
| 16k_bs4_lnc1 | 8 | 8088 |
| 32k_bs1_lnc1 | 9 | 8089 |
| 32k_bs2_lnc1 | 10 | 8090 |
| 32k_bs4_lnc1 | 11 | 8091 |
| 64k_bs1_lnc1 | 12 | 8092 |
| 64k_bs2_lnc1 | 13 | 8093 |
| 64k_bs4_lnc1 | 14 | 8094 |

### Wave 2 (LNC=2): 15 tasks, 2 cores each
| Task | Cores | Port |
|------|-------|------|
| 4k_bs16_lnc2 | 0-1 | 8080 |
| 4k_bs32_lnc2 | 2-3 | 8081 |
| 4k_bs64_lnc2 | 4-5 | 8082 |
| 8k_bs8_lnc2 | 6-7 | 8083 |
| 8k_bs16_lnc2 | 8-9 | 8084 |
| 8k_bs32_lnc2 | 10-11 | 8085 |
| 16k_bs1_lnc2 | 12-13 | 8086 |
| 16k_bs2_lnc2 | 14-15 | 8087 |
| 16k_bs4_lnc2 | 16-17 | 8088 |
| 32k_bs1_lnc2 | 18-19 | 8089 |
| 32k_bs2_lnc2 | 20-21 | 8090 |
| 32k_bs4_lnc2 | 22-23 | 8091 |
| 64k_bs1_lnc2 | 24-25 | 8092 |
| 64k_bs2_lnc2 | 26-27 | 8093 |
| 64k_bs4_lnc2 | 28-29 | 8094 |

## Folder Structure
```
/home/ubuntu/test-bytedance/
  qwen2_1.5b_{len}k_bs{bs}_tp{tp}_lnc{lnc}/
    serve.sh        # vLLM server (cores, port, compile dir hardcoded)
    bench.sh        # vllm bench serve client (matching port)
    compile_cache/  # BASE_COMPILE_WORK_DIR (per-task)
    serve.log       # server stdout/stderr (after run)
    bench.log       # bench stdout/stderr (after run)
  run_all_benchmarks.sh     # orchestrator (parallel waves)
  benchmark_results.log     # aggregated log
  benchmark_results.md      # structured results table
  benchmark_progress.txt    # DONE/FAILED tracking
```

## Key serve.sh Parameters
- `NEURON_RT_VISIBLE_CORES` — which physical cores to use
- `NEURON_RT_NUM_CORES` — how many cores (tp × lnc)
- `BASE_COMPILE_WORK_DIR` — per-task compile cache (avoids conflicts in parallel)
- `--max-num-seqs` = batch_size
- `--max-model-len` = seq_len = token_generation_bucket
- `sequence_parallel_enabled: true` — always on
- Single bucket per task (not multi-bucket)

## Bench Parameters
- `--random-input-len` = input_len (reduced to avoid overflow)
- `--random-output-len` = 300
- `--random-range-ratio` = 0.1
- `--max-concurrency` = batch_size
- `--num-prompts` = 64
- `--request-rate` = inf

## Neuron Core Check Commands
```bash
# Free core count
neuron-ls --json-output | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(d2['nc_count'] for d2 in d) - sum(len(p.get('neuroncore_ids',[])) for d2 in d for p in d2.get('neuron_processes',[])))"

# Which cores are used
neuron-ls --json-output | python3 -c "import sys,json; d=json.load(sys.stdin); used=set(); [used.update(p.get('neuroncore_ids',[])) for d2 in d for p in d2.get('neuron_processes',[])]; print(sorted(used))"
```

## Results Format
| HW | Config Ver. | Dtype | BS | TP | LNC | ITL (ms) | TTFT (s) | E2E (s) | Req/min | Tok/sec |
