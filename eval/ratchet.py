"""Fail if a PR lowers any floor in thresholds.yaml.

    python -m eval.ratchet <(git show origin/main:thresholds.yaml) thresholds.yaml

The one time I lowered a threshold to unblock myself, I spent the next week
not trusting the number. A floor that moves down on demand isn't a floor.
"""
from __future__ import annotations

import sys

import yaml


def main() -> int:
    old = yaml.safe_load(open(sys.argv[1])) or {}
    new = yaml.safe_load(open(sys.argv[2])) or {}
    lowered = {k: (v, new.get(k)) for k, v in old.items() if new.get(k, 0) < v}
    for k, (was, now) in lowered.items():
        print(f"ratchet: {k} lowered {was} -> {now}")
    return 1 if lowered else 0


if __name__ == "__main__":
    sys.exit(main())
