"""
HTTP Server for the Email Triage OpenEnv environment.
Uses FastAPI when available, falls back to stdlib http.server for zero-dep testing.
Exposes step(), reset(), state() as REST endpoints.
"""

import os
import sys
import json
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from email_triage_env import EmailTriageEnv
from models import (
    Action, ActionItem, EmailCategory, Priority,
    Sentiment, Department
)


# ── Session Management ─────────────────────────────────────────────────────

sessions: dict = {}


def get_or_create_session(session_id=None):
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    new_id = session_id or str(uuid.uuid4())[:8]
    sessions[new_id] = EmailTriageEnv()
    return new_id, sessions[new_id]


def parse_action_dict(action_data: dict) -> Action:
    """Parse a raw dict into an Action model."""
    action_items = []
    for item in action_data.get("action_items", []):
        if isinstance(item, dict):
            action_items.append(ActionItem(**item))
        elif isinstance(item, str):
            action_items.append(ActionItem(description=item))

    return Action(
        email_id=action_data.get("email_id", ""),
        category=EmailCategory(action_data.get("category", "support")),
        sentiment=Sentiment(action_data["sentiment"]) if action_data.get("sentiment") else None,
        priority=Priority(action_data["priority"]) if action_data.get("priority") else None,
        department=Department(action_data["department"]) if action_data.get("department") else None,
        action_items=action_items,
        draft_reply=action_data.get("draft_reply"),
        requires_follow_up=action_data.get("requires_follow_up", False),
        notes=action_data.get("notes"),
    )


# ── Try FastAPI first, fallback to stdlib ──────────────────────────────────

def create_fastapi_app():
    """Create the FastAPI application (used in Docker/HF deployment)."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Email Triage OpenEnv",
        description="A real-world email triage environment for AI agents.",
        version="1.0.0",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/")
    def root():
        return {"name": "email-triage", "version": "1.0.0", "status": "running"}

    @app.get("/health")
    def health():
        return {"status": "healthy", "sessions_active": len(sessions)}

    @app.get("/tasks")
    def list_tasks():
        env = EmailTriageEnv()
        return {"tasks": [env.get_task_info(tid) for tid in env.get_task_ids()]}

    @app.post("/reset")
    def reset(req: dict = {}):
        try:
            req = req or {}
            sid, env = get_or_create_session(req.get("session_id"))
            result = env.reset(task_id=req.get("task_id", "task_classify"))
            return {
                "session_id": sid,
                "observation": result.observation.model_dump() if result.observation else None,
                "done": result.done,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/step")
    def step(req: dict):
        sid = req.get("session_id")
        if sid not in sessions:
            raise HTTPException(status_code=404, detail=f"Session '{sid}' not found.")
        env = sessions[sid]
        try:
            action = parse_action_dict(req.get("action", {}))
            result = env.step(action)
            return {
                "observation": result.observation.model_dump() if result.observation else None,
                "reward": result.reward.model_dump(),
                "done": result.done,
                "info": result.info,
            }
        except (RuntimeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/state")
    def get_state(req: dict):
        sid = req.get("session_id")
        if sid not in sessions:
            raise HTTPException(status_code=404, detail=f"Session '{sid}' not found.")
        return sessions[sid].state().model_dump()

    return app


def run_stdlib_server(port=7860):
    """Fallback HTTP server using only stdlib."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, data, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode())

        def _read_body(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length)) if length > 0 else {}

        def do_GET(self):
            if self.path == "/" or self.path == "":
                self._send_json({"name": "email-triage", "version": "1.0.0", "status": "running"})
            elif self.path == "/health":
                self._send_json({"status": "healthy", "sessions_active": len(sessions)})
            elif self.path == "/tasks":
                env = EmailTriageEnv()
                self._send_json({"tasks": [env.get_task_info(tid) for tid in env.get_task_ids()]})
            else:
                self._send_json({"error": "Not found"}, 404)

        def do_POST(self):
            body = self._read_body()
            if self.path == "/reset":
                try:
                    body = body or {}
                    sid, env = get_or_create_session(body.get("session_id"))
                    result = env.reset(task_id=body.get("task_id", "task_classify"))
                    self._send_json({
                        "session_id": sid,
                        "observation": result.observation.model_dump() if result.observation else None,
                        "done": result.done,
                    })
                except ValueError as e:
                    self._send_json({"error": str(e)}, 400)
            elif self.path == "/step":
                sid = body.get("session_id")
                if sid not in sessions:
                    self._send_json({"error": f"Session '{sid}' not found"}, 404)
                    return
                try:
                    action = parse_action_dict(body.get("action", {}))
                    result = sessions[sid].step(action)
                    self._send_json({
                        "observation": result.observation.model_dump() if result.observation else None,
                        "reward": result.reward.model_dump(),
                        "done": result.done,
                        "info": result.info,
                    })
                except (RuntimeError, ValueError) as e:
                    self._send_json({"error": str(e)}, 400)
            elif self.path == "/state":
                sid = body.get("session_id")
                if sid not in sessions:
                    self._send_json({"error": f"Session '{sid}' not found"}, 404)
                    return
                self._send_json(sessions[sid].state().model_dump())
            else:
                self._send_json({"error": "Not found"}, 404)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def log_message(self, format, *args):
            print(f"  [{self.log_date_time_string()}] {format % args}")

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"  Server running at http://0.0.0.0:{port}")
    server.serve_forever()


# ── Main ───────────────────────────────────────────────────────────────────

# Create FastAPI app for uvicorn import
try:
    from fastapi import FastAPI
    app = create_fastapi_app()
except ImportError:
    app = None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    try:
        import uvicorn
        if app:
            print(f"  Starting FastAPI server on port {port}...")
            uvicorn.run(app, host="0.0.0.0", port=port)
        else:
            raise ImportError
    except ImportError:
        print(f"  FastAPI/uvicorn not found, using stdlib server...")
        run_stdlib_server(port)
