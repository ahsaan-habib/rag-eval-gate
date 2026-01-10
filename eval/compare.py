"""Which pairs got worse? The gate says *that* a metric dropped; this says where.

    python -m eval.compare baseline.json results.json
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    args = ap.parse_args()

    base, cand = json.load(open(args.baseline)), json.load(open(args.candidate))
    if base["dataset_version"] != cand["dataset_version"]:
        print(f"warning: comparing {base['dataset_version']} with {cand['dataset_version']}")

    print(f"{'metric':<22} {'base':>7} {'cand':>7} {'delta':>7}")
    for m, b in base["metrics"].items():
        c = cand["metrics"].get(m)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            print(f"{m:<22} {b:7.3f} {c:7.3f} {c - b:+7.3f}")

    b_rows = {r["id"]: r for r in base["rows"]}
    print("\nper-pair changes:")
    for r in cand["rows"]:
        old = b_rows.get(r["id"])
        if not old:
            continue
        notes = []
        if old["refused"] != r["refused"]:
            right = r["refused"] != r["answerable"]
            notes.append(f"{'now refuses' if r['refused'] else 'now answers'} ({'good' if right else 'BAD'})")
        if r.get("context_recall", 1) < old.get("context_recall", 1):
            notes.append(f"lost expected source (ranks {old.get('hit_ranks')} -> {r.get('hit_ranks')})")
        if r.get("faithfulness", 1) < old.get("faithfulness", 1) - 0.2:
            notes.append(f"faithfulness {old['faithfulness']:.2f} -> {r['faithfulness']:.2f}")
        if notes:
            print(f"  {r['id']:<10} " + "; ".join(notes))


if __name__ == "__main__":
    main()
