import json
import sys

import pytest

from eval import compare, gate, ratchet

METRICS = {"context_recall": 0.9, "context_precision": 0.8, "faithfulness": 0.9, "refusal_correctness": 0.9,
           "false_refusal_rate": 0.05, "latency_p50_ms": 900, "latency_p95_ms": 2100}


def results(tmp_path, name="results.json", **override):
    p = tmp_path / name
    rows = [{"id": "a", "answerable": True, "refused": False, "context_recall": 1.0, "faithfulness": 1.0,
             "hit_ranks": [1]},
            {"id": "b", "answerable": False, "refused": True}]
    p.write_text(json.dumps({"dataset": "golden/v3.jsonl", "dataset_version": "v3", "pairs": 2, "model": "m",
                             "prompt": "answer@v3", "metrics": {**METRICS, **override}, "rows": rows}))
    return p


@pytest.fixture
def thresholds(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("context_recall: 0.80\ncontext_precision: 0.75\nfaithfulness: 0.85\nrefusal_correctness: 0.85\n")
    return p


def run(monkeypatch, module, *argv):
    monkeypatch.setattr(sys, "argv", ["x", *map(str, argv)])
    return module.main()


def test_gate_passes_and_fails(tmp_path, thresholds, monkeypatch, capsys):
    assert run(monkeypatch, gate, results(tmp_path), "--thresholds", thresholds) == 0
    assert run(monkeypatch, gate, results(tmp_path, faithfulness=0.7), "--thresholds", thresholds) == 1
    assert "FAIL  faithfulness" in capsys.readouterr().out
    # an explicit flag overrides the file
    assert run(monkeypatch, gate, results(tmp_path), "--thresholds", thresholds, "--min-faithfulness", "0.95") == 1


def test_ratchet_only_goes_up(tmp_path, monkeypatch):
    old, new = tmp_path / "old.yaml", tmp_path / "new.yaml"
    old.write_text("faithfulness: 0.85\ncontext_recall: 0.8\n")
    new.write_text("faithfulness: 0.86\ncontext_recall: 0.8\n")
    assert run(monkeypatch, ratchet, old, new) == 0
    new.write_text("faithfulness: 0.80\n")       # lowered one, dropped the other
    assert run(monkeypatch, ratchet, old, new) == 1


def test_compare_reports_per_pair_changes(tmp_path, monkeypatch, capsys):
    base = results(tmp_path, "base.json")
    cand = json.loads(results(tmp_path, "cand.json").read_text())
    cand["rows"][0].update(refused=True)
    cand["rows"][1].update(refused=False)
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    run(monkeypatch, compare, base, tmp_path / "cand.json")
    out = capsys.readouterr().out
    assert "now refuses (BAD)" in out and "now answers (BAD)" in out
