"""Metrics, one question each.

context recall      did retrieval find an expected passage at all?
context precision   is it near the top?
faithfulness        are the claims supported by what was retrieved?   (judge.py)
refusal correctness does it decline when there is no answer?

Recall and precision are computed from `expected_sources`, not from an LLM
judge — cheaper, deterministic, and it keeps "retrieval failed" separate from
"generation failed".
"""
from __future__ import annotations


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("`", "").split())


def matches(source: str, heading: str, expected: str) -> bool:
    path, _, anchor = expected.partition("#")
    if source != path:
        return False
    return not anchor or _norm(heading) == _norm(anchor)


def hit_ranks(retrieved: list[tuple[str, str]], expected: list[str]) -> list[int]:
    """1-based ranks at which each expected source first appears (missing ones skipped)."""
    ranks = []
    for exp in expected:
        for i, (src, head) in enumerate(retrieved, 1):
            if matches(src, head, exp):
                ranks.append(i)
                break
    return ranks


def context_recall(retrieved: list[tuple[str, str]], expected: list[str]) -> float:
    """Fraction of expected sources present anywhere in the retrieved context."""
    if not expected:
        return 1.0
    return len(hit_ranks(retrieved, expected)) / len(expected)


def context_precision(retrieved: list[tuple[str, str]], expected: list[str]) -> float:
    """Average precision over the ranked context: rewards relevant chunks at the top."""
    if not expected or not retrieved:
        return 0.0
    relevant = [any(matches(s, h, e) for e in expected) for s, h in retrieved]
    hits, total = 0, 0.0
    for k, rel in enumerate(relevant, 1):
        if rel:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def refusal_correctness(rows: list[dict]) -> float:
    """Of the unanswerable pairs, how many did the system decline?"""
    neg = [r for r in rows if not r["answerable"]]
    return mean([1.0 if r["refused"] else 0.0 for r in neg])


def false_refusal_rate(rows: list[dict]) -> float:
    """Of the answerable pairs, how many did it decline? Reported, not gated —
    but a gate that only watches refusal_correctness rewards refusing everything."""
    pos = [r for r in rows if r["answerable"]]
    return mean([1.0 if r["refused"] else 0.0 for r in pos])
