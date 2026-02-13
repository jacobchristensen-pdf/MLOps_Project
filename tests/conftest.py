import sys
from pathlib import Path

# Add repo root to PYTHONPATH so "import src...." works reliably
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))