# rag-eval-gate

The evaluation set and CI regression gate for
[rag-grounded](https://github.com/ahsaan-habib/rag-grounded).

AI regressions don't throw. A worse chunking strategy or a one-word prompt
edit produces fluent, confident, slightly worse answers, and spot-checking
can't see it because you spot-check the questions you already thought about.
This repo turns "it feels better" into four numbers and a red build.

## The golden set

`golden/v3.jsonl` — one pair per line:

```json
{"id": "elo-001",
 "question": "How do I stop a global scope applying to one query?",
 "ground_truth": "Call withoutGlobalScope(ScopeClass::class) ...",
 "expected_sources": ["laravel/eloquent.md#Removing Global Scopes"],
 "answerable": true,
 "note": "phrased as intent, not as the API name — tests semantic retrieval"}
```

- **`expected_sources`** (`path#Heading`) is what makes this more than a quiz:
  retrieval and generation are scored separately, because they fail for
  different reasons and need different fixes.
- **~22% of pairs are deliberately unanswerable**, phrased to sound exactly like
  the answerable ones (`neg-*`). Without them a system that invents answers to
  everything still scores well on every other metric.
- **Versioned, never edited in place.** Scores from v2 and v3 are measured with
  different rulers; `results.json` records which one was used.

Rules I follow when adding pairs: write the question first, then find the
answer in the docs (never the reverse); prefer real questions from chat and
issues over invented ones; mix single-passage, two-section, unanswerable and
ambiguous.

## Metrics

| Metric | Question | Computed from | Floor |
|---|---|---|---|
| context recall | did retrieval find the right passage at all? | `expected_sources` vs reranked chunks | 0.80 |
| context precision | is it near the top? | average precision over the ranking | 0.75 |
| faithfulness | are the answer's claims supported by the retrieved text? | local LLM judge, per claim | 0.85 |
| refusal correctness | does it decline when there's no answer? | refusals on `answerable: false` pairs | 0.85 |

Also reported but not gated: false refusal rate (refusing answerable
questions), p50/p95 latency.

Faithfulness is **grounding, not truth** — it measures agreement with the
retrieved text. If the docs are wrong, a perfectly faithful answer is wrong too.

## Usage

```bash
pip install -e ../rag-grounded -e .     # editable, so prompts/*.yaml resolve
python -m eval.run --dataset golden/v3.jsonl --out results.json
python -m eval.gate results.json            # floors from thresholds.yaml
python -m eval.compare baseline.json results.json   # which pairs moved
```

Runs fully local: the pipeline uses `qwen3:4b-instruct` through Ollama and so does the
judge (`EVAL_JUDGE_MODEL` to change it — a different model family from the
generator is better if you can spare the memory).

## Picking thresholds

Don't guess them.

```bash
python -m eval.noise --dataset golden/v3.jsonl --runs 10
```

runs the unchanged system ten times and prints mean, stdev and a suggested
floor (mean − 4σ). A floor inside the noise band fails on weather; a floor far
below it never fires. Treat it as a ratchet: raise it when the system improves,
never lower it to get a PR through. Floors live in `thresholds.yaml` and CI runs
`eval.ratchet` on PRs, which fails if any floor goes down.

## CI

`.github/workflows/eval.yml` checks out rag-grounded at a ref, builds the
index, runs the set and gates. It runs on PRs here and is a reusable workflow,
so rag-grounded can call it on its own PRs — prompt edits included, since a
prompt change is a behaviour change. `results.json` is uploaded even when the
gate fails.

## What this doesn't catch

Latency and cost regressions (separate gate), tone, adversarial input /
prompt injection via retrieved docs, drift in what users actually ask, and
docs that are themselves out of date.
