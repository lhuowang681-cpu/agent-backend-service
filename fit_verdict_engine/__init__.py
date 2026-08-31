"""Fit Verdict LoRA adapter research package.

Current status: runtime gate PASSED (adapter loads, schema-valid JSON, no
fallback); decision-quality gate REJECTED (heldout v1 adapter accuracy=0.25,
base=0.50). v2 corpora are schema/gate fixtures only, not quality evidence.
Do not start new SFT/GPU training or create v3 corpora.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))


__version__ = "0.1.0"
