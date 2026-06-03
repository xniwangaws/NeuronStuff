#!/usr/bin/env python3
"""Compare Stage 5 canonical validation outputs (neuron vs hf-cpu).

Reads the two JSON files written by stage5_canonical_validation.py and
prints a token-match table plus parsed-response field comparison.

Usage:
    python stage5_compare.py \
        --neuron /home/ubuntu/stage5_neuron_results.json \
        --hf     /home/ubuntu/stage5_hfcpu_results.json \
        --out    /home/ubuntu/agent_artifacts/round4/stage5_comparison.json
"""

import argparse
import json
import sys


def first_n_match(a, b, n):
    """Return (matches, count_compared, pct) over first ``n`` tokens."""
    k = min(n, len(a), len(b))
    if k == 0:
        return 0, 0, 0.0
    matches = sum(1 for i in range(k) if a[i] == b[i])
    return matches, k, 100.0 * matches / k


def parsed_match(p_neuron, p_hf):
    """Exact-string match per parse_response field; tolerate missing keys."""
    if not isinstance(p_neuron, dict) or not isinstance(p_hf, dict):
        return {"_neither_dict": True}
    out = {}
    for k in sorted(set(p_neuron) | set(p_hf)):
        a = p_neuron.get(k)
        b = p_hf.get(k)
        out[k] = {
            "neuron": a,
            "hf":     b,
            "equal":  (a == b),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--neuron", required=True)
    p.add_argument("--hf", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--match-n", type=int, default=16)
    args = p.parse_args()

    with open(args.neuron) as f:
        N = json.load(f)
    with open(args.hf) as f:
        H = json.load(f)

    rows = []
    for prompt, by_thinking in N["by_prompt"].items():
        if prompt not in H["by_prompt"]:
            continue
        for tkey, n_block in by_thinking.items():
            h_block = H["by_prompt"][prompt].get(tkey)
            if h_block is None:
                continue
            for kind in ("greedy", "sample"):
                n_run = n_block.get(kind)
                h_run = h_block.get(kind)
                if not n_run or not h_run:
                    continue
                m, k, pct = first_n_match(n_run["tokens"], h_run["tokens"], args.match_n)
                rows.append({
                    "prompt": prompt,
                    "thinking": tkey,
                    "kind": kind,
                    "match": f"{m}/{k}",
                    "pct": round(pct, 1),
                    "neuron_text": (n_run.get("text_clean") or "")[:80],
                    "hf_text":     (h_run.get("text_clean") or "")[:80],
                    "parsed_match": parsed_match(n_run.get("parsed", {}),
                                                 h_run.get("parsed", {})),
                })

    print(f"{'PROMPT':<10} {'THINKING':<14} {'KIND':<7} {'MATCH':<8} {'PCT':<6}  NEURON / HF")
    for r in rows:
        print(f"{r['prompt']:<10} {r['thinking']:<14} {r['kind']:<7} "
              f"{r['match']:<8} {r['pct']:<6}  "
              f"{r['neuron_text']!r}  ||  {r['hf_text']!r}")

    summary = {
        "match_n": args.match_n,
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
