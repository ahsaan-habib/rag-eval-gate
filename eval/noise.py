"""Measure the noise floor before picking thresholds.

Generation is stochastic, so the same code scores differently run to run.
Run the set N times on unchanged code, look at the spread, and put the gate
well below it — a threshold inside the noise band fails on weather.

    python -m eval.noise --dataset golden/v3.jsonl --runs 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from .gate import GATED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--bands", type=float, default=4.0,
                    help="suggested threshold = mean - bands * stdev")
    ap.add_argument("--dir", default="runs")
    args = ap.parse_args()

    Path(args.dir).mkdir(exist_ok=True)
    scores: dict[str, list[float]] = {m: [] for m in GATED}
    for i in range(args.runs):
        out = Path(args.dir) / f"baseline-{i:02d}.json"
        if not out.exists():  # resumable: a 10-run baseline is slow
            subprocess.run([sys.executable, "-m", "eval.run", "--dataset", args.dataset,
                            "--out", str(out)], check=True)
        metrics = json.loads(out.read_text())["metrics"]
        for m in GATED:
            scores[m].append(metrics[m])

    print(f"{'metric':<22} {'mean':>7} {'stdev':>7} {'min':>7} {'max':>7}  suggested floor")
    for m, xs in scores.items():
        mu = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        floor = max(0.0, mu - args.bands * sd)
        print(f"{m:<22} {mu:7.3f} {sd:7.3f} {min(xs):7.3f} {max(xs):7.3f}  {floor:.2f}")


if __name__ == "__main__":
    main()
