"""Offline by default: a fake embedder stands in for sentence-transformers.
test_smoke_ollama.py talks to a real model and is opt-in (RUN_OLLAMA=1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import fake_st  # noqa: E402

fake_st.install()
