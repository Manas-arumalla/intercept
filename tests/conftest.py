"""Shared test configuration.

Puts the repository root on ``sys.path`` so tests can import the runnable scripts in
``experiments/`` (which is intentionally not an installed package) regardless of how pytest is
invoked (``pytest`` vs ``python -m pytest``).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
