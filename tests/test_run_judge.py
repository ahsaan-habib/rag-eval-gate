from rag_grounded.answer.schemas import Answer, Refusal, Result, Trace
from rag_grounded.ingest.chunker import Chunk
from rag_grounded.llm.ollama import Generation

from eval import judge as judge_mod
from eval.dataset import Pair
from eval.run import evaluate_pair, summarise


class FakeLLM:
    model = "judge"

    def __init__(self, text):
        self.text, self.calls = text, 0

    def chat(self, messages, **kw):
        self.calls += 1
        return Generation(self.text, 10, 5, "judge")


def make_judge(text):
    j = judge_mod.Judge.__new__(judge_mod.Judge)
    j.llm = FakeLLM(text)
    return j


def test_faithfulness_scores_per_claim():
    j = make_judge('{"verdicts": [{"n": 1, "supported": true}, {"n": 2, "supported": false}]}')
    answer = "Call withoutGlobalScope on the query builder [1]. Then restart the whole server twice [1]."
    score, bad = j.faithfulness(answer, ["ctx"])
    assert score == 0.5 and bad == ["Then restart the whole server twice ."]


def test_unparseable_verdict_is_a_fail_not_a_pass():
    j = make_judge("not json at all")
    assert j.faithfulness("Call withoutGlobalScope on the query builder [1].", ["ctx"])[0] == 0.0
    assert make_judge("x").faithfulness("Ok.", ["ctx"]) == (1.0, [])  # no claims, no call


class FakePipe:
    class retriever:
        class vectors:
            @staticmethod
            def get(ids):
                return [Chunk(i, "laravel/eloquent.md", "Removing Global Scopes", "t") for i in ids]

    def __init__(self, out):
        self.out = out

    def ask(self, q):
        return Result(output=self.out, trace=Trace(query=q, reranked=[("c1", 3.0)], contexts=["t"],
                                                   input_tokens=100, output_tokens=20))


def test_evaluate_pair_and_summary():
    pos = Pair(id="p", question="q", ground_truth="g", expected_sources=["laravel/eloquent.md#Removing Global Scopes"],
               answerable=True)
    neg = Pair(id="n", question="q", ground_truth=None, expected_sources=[], answerable=False)
    j = make_judge('{"verdicts": [{"n": 1, "supported": true}]}')
    r1 = evaluate_pair(FakePipe(Answer(text="Use withoutGlobalScope on the builder [1].", citations=[])), j, pos)
    r2 = evaluate_pair(FakePipe(Refusal(message="no", reason="no_context")), j, neg)
    assert r1["context_recall"] == 1.0 and r1["faithfulness"] == 1.0 and r1["hit_ranks"] == [1]
    assert r2["refused"] and r2["refusal_reason"] == "no_context"
    m = summarise([r1, r2])
    assert m["refusal_correctness"] == 1.0 and m["false_refusal_rate"] == 0.0 and m["faithfulness"] == 1.0
