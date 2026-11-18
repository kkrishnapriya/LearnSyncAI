"""
LearnSync AI — Source Package
Provides clean imports for all three member modules.
"""
import sys
import os

# Ensure data/ (Member 1) and Member_2/ and Member_3/ are on the path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PATHS = [
    os.path.join(_ROOT, "data"),          # Member 1: assessment, bkt_engine, data_pipeline
    os.path.join(_ROOT, "Member_2"),      # Member 2: rag, evaluator
    os.path.join(_ROOT, "Member_3"),      # Member 3: database, curriculum, orchestrator
]

for _p in _PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)
