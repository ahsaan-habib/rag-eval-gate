"""python -m eval.gate results.json --min-faithfulness 0.85 ...

Exits non-zero on any breach -> the build fails -> the change does not merge.
"""
from __future__ import annotations

import argparse
import json
import sys

GATED = ["context_recall", "context_precision", "faithfulness", "refusal_correctness"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    for m in GATED:
        ap.add_argument(f"--min-{m.replace('_', '-')}", type=float, required=True, dest=m)
    args = ap.parse_args()

    res = json.load(open(args.results))
    got = res["metrics"]
    failed = False
    print(f"{res['dataset']} ({res['pairs']} pairs), {res['model']}, prompt {res['prompt']}\n")
    for m in GATED:
        floor, value = getattr(args, m), got[m]
        ok = value >= floor
        failed |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {m:<22} {value:.3f}  (min {floor:.2f})")
    for m in ("false_refusal_rate", "latency_p50_ms", "latency_p95_ms"):
        print(f"  info  {m:<22} {got[m]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
