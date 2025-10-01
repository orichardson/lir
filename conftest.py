"""Pytest configuration for local imports.

Ensures the repository's `code/` directory is placed at the front of
`sys.path` so that imports like `code.*` resolve to this project rather than
the Python stdlib module named `code`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure both possible project roots are importable:
# - code/ contains modules like `lir__simpler.py`
# - lir/ now contains tests; keeping it on path is harmless and can help
PKG_DIR = PROJECT_ROOT / "lir"
sys.path.insert(0, str(PKG_DIR))
