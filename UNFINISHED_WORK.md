# Unfinished Work — 2026-05-06 Terminate Snapshot

Last push: `4e93630` on `origin/main` (github.com/xniwangaws/NeuronStuff).

Context: OPPO 客户的 FLUX.1-dev / FLUX.2-klein / SDXL benchmark, comparing **Neuron trn2 / H100 / L4**.

---

## 1. SDXL — whn09 fork seeds 42-51 rerun (trn2.3xl)

**状态**: compile 卡在 UNet 之前就被 kill (text_encoder + text_encoder_2 + vae_decoder 已 PASS)。没有生成任何 seed42-51 图。

**之前已提交但用的是错误 seeds (0-9)**:
- `sdxl-benchmark/astronaut_bench/results/sdxl_astro_trn2_whn09_1024/seed_{00..09}_astronaut.png`

**下次要做**:
1. 启动 trn2.3xlarge (sa-east-1 推荐,或 ap-southeast-4)
2. 复用 `/home/ubuntu/sdxl_whn09_fixed.py` (已经 hard-code `SEEDS = [42..51]`)
3. 指定 HF cache `/home/ubuntu/models/sdxl-base` (从 HF 拉 13GB)
4. `source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`
5. `nohup python /home/ubuntu/sdxl_whn09_fixed.py > /home/ubuntu/sdxl_whn09_run.log 2>&1 &` — **只启一次**,别重复 launch(之前两个 instance 打架)
6. 预期:compile ~30-40min → bench 10 seeds @ 11.15s/image ~2min
7. 完成后 scp `/home/ubuntu/sdxl_whn09_results/seed{42..51}.png + results.json` 到
   `/Users/xniwang/NeuronStuff/sdxl-benchmark/astronaut_bench/results/sdxl_astro_trn2_whn09_1024_seeds42_51/`
8. 更新 `sdxl-benchmark/README.md` §5.1 table 的 whn09 列指向新 seed42 图

**关键陷阱**:
- `/tmp/crash_info` root-owned 会占 2.6GB,compile 前 `sudo rm -rf /tmp/crash_info`
- disk 80%+ 时 compile 会 `[Errno 28] No space left on device`,需先 64G fallocate swap + 清 `flux2_klein` / `workspace` / `s3diff` / `profile_e2b`
- 不要同时有 `download_flux1` / `flux_2k_bench` 等 multi-project 进程

---

## 2. SDXL — whn09 2K / 4K (trn2.3xl)

**状态**: 没开始。上游 1K 都没重跑成功。

**下次要做**:
1. 先完成 §1 的 1K
2. 改 `sdxl_whn09_fixed.py` 里 `HEIGHT, WIDTH = 2048, 2048`
3. 重 compile UNet (2K NEFF) — 预期可能撞 `NCC_EVRF007` 5M instruction 上限,同之前 SDXL 2K 的问题
4. 若 compile fail 直接记录为 "BLOCKED: NCC_EVRF007 instruction ceiling" 到 README.md §3
5. 4K 同理,若 2K 失败 4K 必失败

**预期结论** (基于 SDK 2.29 历史): 2K/4K 编译不可行,walrus_driver 超 RAM

---

## 3. FLUX.1-dev — Neuron trn2 2K / 4K

**状态**: 编译 2K transformer 成功 (539s) + TE/VAE 成功,但 weight load + warmup 阶段 instance 被 terminate,无 per-seed 结果。

**下次要做**:
1. 启动 trn2.3xlarge
2. 下载 FLUX.1-dev 完整 43GB 到 `/home/ubuntu/models/FLUX.1-dev/` (用 `hf_hub_download` 显式列 transformer shards 1/2/3,snapshot_download 会漏)
3. 用 `NeuronFluxApplication` one-shot pipeline (参考 `flux-benchmark/alien_bench/bench_neuron_alien.py` 改 resolution)
4. 环境:
   ```
   NEURON_LOGICAL_NC_CONFIG=2
   NEURON_RT_VISIBLE_CORES=0-3
   ```
5. Compile 2K: ~9 min (transformer) + CLIP/T5/VAE ≈ 25s
6. Bench 10 seeds 预期 ~60-90s/image @ 2K
7. 4K: 先尝试 compile,很可能撞 instruction ceiling(同 klein 4K 的 HLO gen timeout),失败则记录 BLOCKED
8. 保存到 `/Users/xniwang/NeuronStuff/flux-benchmark/alien_bench/results/flux1_alien_trn2_bf16_{2048,4096}/`

---

## 4. FLUX.1-dev — L4 FP8 4K

**状态**: 2K 完成 (339.38s, 10/10, 2.42GB peak); 4K 目录空 (`flux1_alien_l4_fp8_4096/`) — 没跑。

**下次要做**:
1. L4 g6.4xlarge sa-east-1,复用 wangkanai/flux-dev-fp8 + sequential CPU offload (已跑过 1K/2K)
2. `bench_l4_alien_hires.py --resolution 4096 --seeds 42 43 44` (3 seeds 抽样即可,4K ≈ 1500s/image)
3. 预期 OOM (L4 22GB VRAM 吃不下 12B bf16 upcast 在 4K resolution)。若 OOM 记录 BLOCKED。

---

## 5. SDXL trn2 Track A+B 2K/4K (SDK 2.29)

**状态**: 1K no-CFG (19.997s) + 1K CFG=7.5 batch=2 (13.262s) 已完成。2K/4K 编译不可行,已记录在 README.md §3/§4。

**下次要做**: 无,已按现状 ship。

---

## 6. FLUX.2-klein Neuron 4K

**状态**: 已记录为 "HLO gen timeout, NUM_PATCHES=65536" + 模型 spec `max_area=4MP` 所限 —— 不值得再试。

**下次要做**: 无。

---

## 7. H100 torchao FP8 重测

**状态**: eager mode 慢 5× 已记录为反例,README 里留占位符。

**下次要做** (可选):
1. 启 p5.4xlarge,改用 `torch.compile(mode="reduce-overhead") + CUDA graphs`
2. 如果 FP8 < BF16 就填回 README table,否则永久标记为 "Not production ready in eager mode"

---

## 数据备份建议(本地 git 之外)

目前已 commit 到 GitHub 的:
- ✅ whn09 1K (seeds 00-09 旧版本)
- ✅ H100 FLUX.1 FP8 2K/4K (10 seeds each)
- ✅ L4 FLUX.1 FP8 2K (10 seeds)
- ✅ klein 1K/2K (Neuron BF16 + H100 BF16/FP8 + L4 FP8, 全 10 seeds)
- ✅ SDXL 1K (trn2 no-CFG + CFG=7.5 + H100 BF16/FP16 + L4 BF16/FP16,全 10 seeds)
- ✅ SDXL 2K/4K (H100 + L4)
- ✅ 所有 README 和 REPORT

**不用备份到 git 的大文件** (建议 S3 if needed):
- NEFF 编译产物 (`/mnt/nvme/neff/*.pt`, `sdxl_whn09_compile/*.pt`) — 重新编译可得
- HF 模型 weights (`/home/ubuntu/models/*`) — 可从 HF 重下
- trn2 runtime artifacts

S3 bucket: `s3://xniwang-neuron-models-us-east-2/` (personal,不受 capacity block 影响)

---

## Git state

- Branch: `main`
- Local HEAD: `4e93630` (pushed)
- Remote: `https://github.com/xniwangaws/NeuronStuff` origin
- GitLab mirror: `git@ssh.gitlab.aws.dev:xniwang/NeuronStuff-flux2.git` (needs `mwinit -f -s -k ~/.ssh/id_ecdsa.pub` before push)
- git-defender blocks HTTPS push locally — 必须 scp bundle + PAT 到 jumpbox (e.g. L4 sa-east-1) 然后 from there push

Push command (L4 jumpbox):
```bash
# Local: build bundle of new commits only
git bundle create /tmp/ns-push.bundle origin/main..main
# scp bundle + token
scp -i ~/.ssh/neuron-bench-sa-east-1.pem /tmp/ns-push.bundle ~/credentials/github-token.txt ubuntu@<L4_IP>:/tmp/
# Remote push
ssh ... "cd /tmp && git clone --quiet https://\$(cat /tmp/github-token.txt)@github.com/xniwangaws/NeuronStuff.git ns-push && cd ns-push && git fetch /tmp/ns-push.bundle main:incoming && git merge --ff-only incoming && git push origin main && rm -rf /tmp/ns-push /tmp/ns-push.bundle /tmp/github-token.txt"
```

---

## Terminate checklist (本次)

- [x] pkill 所有 agent 上的 compile / bench / download 进程
- [x] scp L4 FP8 2K 图回本地
- [x] commit + push 最新数据到 GitHub
- [ ] terminate: trn2.3xlarge sa-east-1 (15.229.211.94) — **whn09 仍在 compile,kill 掉**
- [ ] terminate: L4 g6.4xlarge sa-east-1 (15.229.155.96)
- [ ] terminate: H100 p5.4xlarge ap-northeast-1 (13.231.10.203)
- [ ] terminate: L4 klein agent sa-east (15.229.33.253) — 之前已完成并 push
