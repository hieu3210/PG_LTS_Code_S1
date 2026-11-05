"""Reviewer experiments; adds the bundled hedge-algebra core to Python's path."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party' / 'pyLTS'))
__version__ = '1.0.0'
