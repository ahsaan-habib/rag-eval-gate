"""The judge against a real local model. Opt-in:

    RUN_OLLAMA=1 pytest tests/test_smoke_ollama.py -s
"""
import os

import pytest

from eval.judge import Judge

pytestmark = pytest.mark.skipif(os.environ.get("RUN_OLLAMA") != "1", reason="set RUN_OLLAMA=1 to run")


def test_judge_separates_supported_from_invented():
    ctx = ["If you would like to remove a global scope for a given query, you may use the "
           "withoutGlobalScope method, passing the class name of the scope."]
    answer = ("Use the withoutGlobalScope method to remove a global scope for a single query [1]. "
              "Global scopes are cached in Redis for exactly ten minutes by default [1].")
    score, unsupported = Judge().faithfulness(answer, ctx)
    print("\nscore", score, "unsupported", unsupported)
    assert score == 0.5 and "Redis" in unsupported[0]
