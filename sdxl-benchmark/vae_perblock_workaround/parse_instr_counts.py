"""Parse HLO instruction counts and per-block compile stats from compile_<name> dirs."""
import json
import re
from pathlib import Path

WORK = Path("/home/ubuntu/work_a")

PATTERNS = [
    re.compile(r"(\d[\d,]*)\s+HLO\s+instruction", re.IGNORECASE),
    re.compile(r"instruction[_\s]*count[^\d]*([\d,]+)", re.IGNORECASE),
    re.compile(r"number\s+of\s+instructions[^\d]*([\d,]+)", re.IGNORECASE),
    re.compile(r"total\s+instructions[^\d]*([\d,]+)", re.IGNORECASE),
]

def parse_instr(workdir):
    candidates = []
    for log in workdir.rglob("*"):
        if not log.is_file():
            continue
        if log.suffix not in (".log", ".txt", ".json"):
            continue
        try:
            txt = log.read_text(errors="ignore")
        except Exception:
            continue
        for pat in PATTERNS:
            for m in pat.finditer(txt):
                v = int(m.group(1).replace(",", ""))
                candidates.append((v, str(log), m.group(0)[:120]))
    return candidates

results = {}
for d in sorted(WORK.glob("compile_*")):
    name = d.name.replace("compile_", "")
    hits = parse_instr(d)
    if hits:
        # take the largest unique value seen (HLO instr count is usually the biggest figure)
        unique = sorted(set((v, src) for v, src, _ in hits), reverse=True)
        results[name] = {
            "max_instr_seen": unique[0][0],
            "source": unique[0][1],
            "all_top": [v for v, _ in unique[:5]],
        }
    else:
        results[name] = {"max_instr_seen": None}

# Also collect compile-dir size as a proxy.
for d in sorted(WORK.glob("compile_*")):
    name = d.name.replace("compile_", "")
    total = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    results.setdefault(name, {})["compile_dir_bytes"] = total

# Add traced .pt size and trace_log entries.
log = json.loads((WORK / "trace_log.json").read_text())
for name, entry in log.items():
    results.setdefault(name, {}).update(entry)

(WORK / "instr_counts.json").write_text(json.dumps(results, indent=2, sort_keys=True))
for name in sorted(results.keys()):
    r = results[name]
    print(f"{name:24s} instr={str(r.get('max_instr_seen')):>12s}  "
          f"compile_s={r.get('elapsed_s')}  workdir_MB={r.get('compile_dir_bytes', 0)/1e6:.1f}")
print(f"\nWrote {WORK / 'instr_counts.json'}")
