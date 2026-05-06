# Trn2 `/tmp/` artifact backup (sa-east-1 instance before termination)

Misc artifacts from `/tmp/` on the Trn2 dev box that were generated during
Phase E/R bisection + bus benchmarking but never made it into the main
`customer_report/` directory. Preserved here for reproducibility and audit.

## Files

### Bisect trial outputs (Trial 1-7, Phase R4c follow-up)

- `bisect_trial_{1..7}.log` — stdout per-trial at 1K (cat)
- `bisect_summary.md` — Phase R bisect summary table (also in `src/data/`)

Note: trial output PNGs are already in `customer_report/images/` (named
`bisect_trial_{1..6}.png` copied from phase_r during promote).

### Bus benchmark (last run before instance termination)

Actual bus series runs done on 2026-05-06 ~01:00-01:25 UTC, used to replace
cat images in the customer_report:

- `bus_LQ_{256,512,1024}.png` — BICUBIC-downsampled bus inputs from
  `https://ultralytics.com/images/bus.jpg`
- `bus1k.log` / `bus2k.log` / `bus4k.log` — per-resolution run logs
- `bus_seq.log` — wrapper script output (1K → 2K → 4K sequence)
- `trial6_bus_1k.png` / `_2k.png` / `_4k.png` — produced images (also in
  `customer_report/images/`)
- `trial6_bus_1k_real.png` — earlier attempt with correct (real) bus input

## How these got here

The sa-east-1 `trn2.3xlarge` instance's `~/workspace/` directory was
unexpectedly cleared between 2026-05-05 and 2026-05-06 (possibly by a
disk-reclaim script or manual cleanup by another agent). The `/tmp/`
directory survived and these are the salvaged outputs. All scripts that
produced them are in `backup/phase_r/scripts/` and `src/scripts/`.
