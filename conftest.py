"""Pytest configuration for local imports.

Ensures the repository's `code/` directory is placed at the front of
`sys.path` so that imports like `code.*` resolve to this project rather than
the Python stdlib module named `code`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "lir"
sys.path.insert(0, str(CODE_DIR))
