"""
server/app.py — OpenEnv multi-mode deployment entry point.
Re-exports the FastAPI app and main entry point from the server package.
"""

from server import app, main  # noqa: F401

__all__ = ["app", "main"]
