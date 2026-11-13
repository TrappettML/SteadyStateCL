"""Make ``sscl_core`` (the shared package, one level up) importable when a script is run from anywhere.

``ROOT`` is this experiment's folder (its ``configs/`` and ``results/``); ``REPO`` is the folder that holds
``sscl_core/`` and every experiment.  Each experiment's ``scripts/_path.py`` is this same file.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
for _p in (ROOT, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
