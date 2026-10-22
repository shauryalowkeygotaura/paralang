"""
live.py -- a local feed of live calls for a supervisor view (v0.3 #12). STDLIB ONLY.

    feed = LiveFeed.start()                       # 127.0.0.1:8765, once per process
    feed.publish(call_id, {"turn": 3, "tag": "[tone: hesitant]", ...})
    feed.end(call_id, {"tier": "warm", "score": 55})

    GET http://127.0.0.1:8765/calls   -> {"calls": [...], "ts": ...}
    GET http://127.0.0.1:8765/health  -> {"ok": true}

Command Center's LIVE CALLS panel polls /calls from the browser. Privacy by
construction: the server binds to loopback only (it refuses any other host), so
call data never leaves the machine running the agent; the page is just a viewer
that happens to be served from GitHub Pages.

Browser rules this has to satisfy: a public https page fetching http://127.0.0.1
is allowed (localhost counts as a secure context), but needs CORS for the page's
origin (ALLOWED_ORIGINS, extend with PARALANG_LIVE_ORIGINS) and Chrome's Private
Network Access preflight header (Access-Control-Allow-Private-Network: true).
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PARALANG_LIVE_PORT", "8765"))
ALLOWED_ORIGINS = {"https://shauryalowkeygotaura.github.io", "http://localhost:3000",
                   "http://127.0.0.1:3000"} | {o.strip() for o in os.environ.get(
                       "PARALANG_LIVE_ORIGINS", "").split(",") if o.strip()}
KEEP_ENDED_S = 600          # an ended call stays visible for 10 minutes
MAX_TURNS = 40              # per call, newest kept
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class LiveFeed:
    _instance: "LiveFeed | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self.calls: dict[str, dict] = {}
        self._data_lock = threading.Lock()
        self.server: ThreadingHTTPServer | None = None
        self.port: int | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    @classmethod
    def start(cls, port: int = PORT, host: str = "127.0.0.1") -> "LiveFeed":
        """The process-wide feed, started on first call. A busy port leaves the
        feed in-memory only (agents must never fail because a viewer is off)."""
        if host not in _LOOPBACK:
            raise ValueError(f"live feed binds to loopback only, not {host!r}")
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            feed = cls()
            try:
                feed.server = ThreadingHTTPServer((host, port), _handler_for(feed))
                feed.port = feed.server.server_address[1]
                threading.Thread(target=feed.server.serve_forever, daemon=True,
                                 name="paralang-live").start()
            except OSError:
                feed.server = None                  # port taken: keep publishing in memory
            cls._instance = feed
            return feed

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        with LiveFeed._lock:
            if LiveFeed._instance is self:
                LiveFeed._instance = None

    # ── data ───────────────────────────────────────────────────────────────
    def publish(self, call_id: str, event: dict, meta: dict | None = None) -> None:
        now = time.time()
        with self._data_lock:
            c = self.calls.setdefault(call_id, {"id": call_id, "started": now, "status": "live",
                                                "turns": [], "meta": {}})
            if meta:
                c["meta"].update(meta)
            c["turns"] = (c["turns"] + [{**event, "t": round(now, 2)}])[-MAX_TURNS:]
            c["updated"] = now

    def end(self, call_id: str, result: dict | None = None) -> None:
        with self._data_lock:
            c = self.calls.get(call_id)
            if c:
                c["status"], c["result"], c["updated"] = "ended", result or {}, time.time()

    def snapshot(self) -> dict:
        now = time.time()
        with self._data_lock:
            for k in [k for k, c in self.calls.items()
                      if c["status"] == "ended" and now - c["updated"] > KEEP_ENDED_S]:
                del self.calls[k]
            # default=str: one odd value in an event must not take /calls down
            calls = sorted((json.loads(json.dumps(c, default=str)) for c in self.calls.values()),
                           key=lambda c: (c["status"] != "live", -c["updated"]))
        return {"calls": calls, "ts": round(now, 2)}


def _handler_for(feed: LiveFeed):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the agent's console clean
            pass

        def _cors(self):
            origin = self.headers.get("Origin", "")
            if origin in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _host_ok(self) -> bool:
            """DNS-rebinding guard: a hostile page can point ITS domain at
            127.0.0.1 and read us same-origin, bypassing CORS. Loopback binding
            alone does not stop that; insisting on a localhost Host header does."""
            host = (self.headers.get("Host") or "").strip().lower()
            name = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.split(":")[0]
            if name in _LOOPBACK:
                return True
            body = b'{"error": "forbidden host"}'
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False

        def do_OPTIONS(self):
            if not self._host_ok():
                return
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_GET(self):
            if not self._host_ok():
                return
            path = self.path.split("?", 1)[0]
            if path == "/calls":
                body, code = json.dumps(feed.snapshot(), default=str).encode(), 200
            elif path == "/health":
                body, code = b'{"ok": true}', 200
            else:
                body, code = b'{"error": "not found"}', 404
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def turn_event(res, coach: str | None = None, lang: str | None = None) -> dict:
    """A TurnResult as the small, viewer-safe event the panel shows."""
    from .report import engagement
    d = res.to_dict()
    return {"turn": d["turn"], "tag": d["tag"], "text": (d["text"] or "")[:80],
            "cues": [{"name": c["name"], "strength": c["strength"]} for c in d["cues"]],
            "engagement": round(engagement(d)), "coach": coach, "lang": lang}
