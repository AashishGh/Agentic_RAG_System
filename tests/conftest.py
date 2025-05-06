# tests/conftest.py
import sys, os

# ensure project root is on path so `src` is recognized as a package
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
