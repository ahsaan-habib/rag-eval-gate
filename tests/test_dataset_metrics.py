from pathlib import Path

import pytest

from eval import dataset, metrics

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


@pytest.mark.parametrize("path", sorted(GOLDEN.glob("v*.jsonl")), ids=lambda p: p.name)
def test_every_golden_set_validates(path):
    pairs = dataset.load(path)
    assert pairs and any(not p.answerable for p in pairs)
    for p in pairs:
        if p.answerable:
            assert all("#" in s or s.endswith(".md") for s in p.expected_sources), p.id


def test_dataset_rejects_inconsistent_and_duplicate_pairs(tmp_path):
    bad = tmp_path / "v9.jsonl"
    bad.write_text('{"id": "x", "question": "q", "ground_truth": null, "expected_sources": [], "answerable": true}\n')
    with pytest.raises(ValueError, match="answerable pair needs"):
        dataset.load(bad)
    line = '{"id": "x", "question": "q", "ground_truth": null, "expected_sources": [], "answerable": false}\n'
    bad.write_text(line * 2)
    with pytest.raises(ValueError, match="duplicate ids"):
        dataset.load(bad)
    assert dataset.version_of("golden/v3.jsonl") == "v3" and dataset.version_of("x/custom.jsonl") == "custom"


RETRIEVED = [("laravel/queues.md", "Jobs"), ("laravel/eloquent.md", "Removing `Global` Scopes"),
             ("laravel/eloquent.md", "Soft Deleting")]


def test_source_matching_and_ranks():
    assert metrics.matches("laravel/eloquent.md", "Removing Global Scopes", "laravel/eloquent.md#removing global scopes")
    assert metrics.matches("laravel/eloquent.md", "anything", "laravel/eloquent.md")
    assert not metrics.matches("laravel/queues.md", "Jobs", "laravel/eloquent.md")
    expected = ["laravel/eloquent.md#Removing Global Scopes", "laravel/routing.md#Basics"]
    assert metrics.hit_ranks(RETRIEVED, expected) == [2]
    assert metrics.context_recall(RETRIEVED, expected) == 0.5
    assert metrics.context_recall(RETRIEVED, []) == 1.0


def test_context_precision_rewards_the_top():
    exp = ["laravel/eloquent.md#Removing Global Scopes"]
    assert metrics.context_precision(RETRIEVED, exp) == pytest.approx(0.5)
    assert metrics.context_precision(list(reversed(RETRIEVED)), exp) == pytest.approx(0.5)
    assert metrics.context_precision([RETRIEVED[1]], exp) == 1.0
    assert metrics.context_precision([], exp) == 0.0


def test_refusal_metrics_are_kept_apart():
    rows = [{"answerable": False, "refused": True}, {"answerable": False, "refused": False},
            {"answerable": True, "refused": True}, {"answerable": True, "refused": False}]
    assert metrics.refusal_correctness(rows) == 0.5
    assert metrics.false_refusal_rate(rows) == 0.5
    assert metrics.mean([]) == 0.0
