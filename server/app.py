"""
server/app.py — OpenEnv multi-mode deployment entry point.
This file is the canonical entry point required by the OpenEnv spec.
"""

import os
import sys

# Add project root to path so sibling modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: F401 — re-export FastAPI app for uvicorn


def main():
    """
    Start the Email Triage OpenEnv HTTP server.
    Called by the [project.scripts] entry point: serve = 'server.app:main'
    """
    port = int(os.environ.get("PORT", 7860))
    try:
        import uvicorn
        if app:
            print(f"Starting Email Triage OpenEnv server on port {port}...")
            uvicorn.run(app, host="0.0.0.0", port=port)
        else:
            raise ImportError("FastAPI app not available")
    except ImportError:
        # Fallback to stdlib server
        from server import run_stdlib_server
        print(f"FastAPI/uvicorn not available, using stdlib server on port {port}...")
        run_stdlib_server(port)


if __name__ == "__main__":
    main()
