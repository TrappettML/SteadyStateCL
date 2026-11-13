"""Old name of the shared package: ``mup`` moved to ``sscl_core/`` (one level up) on 2026-11-13.

``import mup`` and ``from mup.model import …`` still work and give the *same* module objects as
``sscl_core`` and ``sscl_core.model``; new code should import ``sscl_core``.
"""
import importlib
import importlib.abc
import importlib.util
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import sscl_core  # noqa: E402

__version__ = sscl_core.__version__


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """``mup.<name>`` → the module ``sscl_core.<name>`` itself (not a copy)."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("mup."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("sscl_core." + spec.name[len("mup."):])

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Alias())
