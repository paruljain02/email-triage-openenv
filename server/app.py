"""
server/app.py — OpenEnv multi-mode deployment entry point.
Imports and re-exports the FastAPI app from the root server.py.
"""

import sys
import os

# Add project root to path so root-level modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app, main  # noqa: F401

__all__ = ["app", "main"]
