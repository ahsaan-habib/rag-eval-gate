"""python -m eval.run --dataset golden/v3.jsonl --out results.json"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from rag_grounded.pipeline import RAGPipeline

from . import dataset, metrics
from .judge import Judge


def evaluate_pair(pipe: RAGPipeline, judge: Judge, pair: dataset.Pair) -> dict:
    t0 = time.perf_counter()
    res = pipe.ask(pair.question)
    latency = (time.perf_counter() - t0) * 1000

    ranked_ids = [cid for cid, _ in res.trace.reranked]
    chunks = pipe.retriever.vectors.get(ranked_ids)
    retrieved = [(c.source, c.heading) for c in chunks]

    row = {
        "id": pair.id,
        "answerable": pair.answerable,
        "refused": res.refused,
        "refusal_reason": getattr(res.output, "reason", None),
        "retrieved": [f"{s}#{h}" for s, h in retrieved],
        "latency_ms": round(latency, 1),
        "input_tokens": res.trace.input_tokens,
        "output_tokens": res.trace.output_tokens,
    }
    if pair.answerable:
        row["hit_ranks"] = metrics.hit_ranks(retrieved, pair.expected_sources)
        row["context_recall"] = metrics.context_recall(retrieved, pair.expected_sources)
        row["context_precision"] = metrics.context_precision(retrieved, pair.expected_sources)
        if not res.refused:
            score, unsupported = judge.faithfulness(res.output.text, res.trace.contexts)
            row["faithfulness"] = round(score, 4)
            row["unsupported_claims"] = unsupported
            row["answer"] = res.output.text
    return row


def summarise(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["answerable"]]
    answered = [r for r in pos if "faithfulness" in r]
    lat = sorted(r["latency_ms"] for r in rows)
    pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else 0.0  # noqa: E731
    return {
        "context_recall": round(metrics.mean([r["context_recall"] for r in pos]), 4),
        "context_precision": round(metrics.mean([r["context_precision"] for r in pos]), 4),
        "faithfulness": round(metrics.mean([r["faithfulness"] for r in answered]), 4),
        "refusal_correctness": round(metrics.refusal_correctness(rows), 4),
        "false_refusal_rate": round(metrics.false_refusal_rate(rows), 4),
        "latency_p50_ms": pct(0.50),
        "latency_p95_ms": pct(0.95),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only (smoke runs)")
    args = ap.parse_args()

    pairs = dataset.load(args.dataset)
    if args.limit:
        pairs = pairs[: args.limit]
    pipe, judge = RAGPipeline(), Judge()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(lambda p: evaluate_pair(pipe, judge, p), pairs))

    out = {
        "dataset": args.dataset,
        "dataset_version": dataset.version_of(args.dataset),
        "pairs": len(rows),
        "unanswerable": sum(not r["answerable"] for r in rows),
        "model": pipe.llm.model,
        "judge_model": judge.llm.model,
        "prompt": f"{pipe.prompt.id}@v{pipe.prompt.version}",
        "metrics": summarise(rows),
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
