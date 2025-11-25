from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, model_validator


class Pair(BaseModel):
    id: str
    question: str
    ground_truth: str | None
    expected_sources: list[str]   # "path/to/file.md#Heading"
    answerable: bool
    note: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        if self.answerable and not (self.ground_truth and self.expected_sources):
            raise ValueError(f"{self.id}: answerable pair needs ground_truth and expected_sources")
        if not self.answerable and (self.ground_truth or self.expected_sources):
            raise ValueError(f"{self.id}: unanswerable pair must have no ground_truth/sources")
        return self


def load(path: str | Path) -> list[Pair]:
    pairs = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                pairs.append(Pair.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"{path}:{n}: {e}") from e
    ids = [p.id for p in pairs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate ids: {sorted(dupes)}")
    return pairs


def version_of(path: str | Path) -> str:
    """golden/v3.jsonl -> v3. Scores from different versions are not comparable."""
    m = re.search(r"(v\d+)", Path(path).stem)
    return m.group(1) if m else Path(path).stem
