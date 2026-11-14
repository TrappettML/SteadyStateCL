import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "scripts"), ROOT, os.path.dirname(ROOT)):   # the scripts, the experiment, the repo root (sscl_core/)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# On a GPU node the test process must not grab 75 % of every card.  Must precede the first ``import jax``.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
