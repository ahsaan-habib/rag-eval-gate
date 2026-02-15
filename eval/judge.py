"""Faithfulness judge: what fraction of an answer's claims are supported by
the retrieved context? Same definition Ragas uses, run against a local model
so the eval costs nothing and needs no API key.

Caveat worth knowing: if the judge and the generator are the same model, the
judge is a little too forgiving of its own phrasing. Set EVAL_JUDGE_MODEL to a
different family when you can afford the extra pull.
"""
from __future__ import annotations

import json
import os

from pydantic import BaseModel, ValidationError

from rag_grounded.answer.claims import split_claims, strip_citations
from rag_grounded.llm.ollama import Ollama

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "qwen3:4b-instruct")

SYSTEM = """You check whether statements are supported by context passages.
A statement is supported only if the passages state it or directly imply it.
Background knowledge does not count. Reply in JSON."""

USER = """Context passages:
{context}

Statements:
{claims}

For each statement number, decide supported true/false."""


class Verdict(BaseModel):
    n: int
    supported: bool


class Verdicts(BaseModel):
    verdicts: list[Verdict]


class Judge:
    def __init__(self, model: str = JUDGE_MODEL):
        self.llm = Ollama(model=model, timeout=120)

    def faithfulness(self, answer: str, contexts: list[str]) -> tuple[float, list[str]]:
        claims = [strip_citations(c) for c in split_claims(answer)]
        if not claims:
            return 1.0, []
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(
                context="\n\n".join(f"<<{i}>>\n{c}" for i, c in enumerate(contexts, 1)),
                claims="\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1)),
            )},
        ]
        raw = self.llm.chat(messages, fmt=Verdicts.model_json_schema()).text
        try:
            verdicts = {v.n: v.supported for v in Verdicts.model_validate_json(raw).verdicts}
        except (ValidationError, json.JSONDecodeError):
            # an unparseable verdict counts as unsupported, never as a pass
            return 0.0, claims
        unsupported = [c for i, c in enumerate(claims, 1) if not verdicts.get(i, False)]
        return 1 - len(unsupported) / len(claims), unsupported
