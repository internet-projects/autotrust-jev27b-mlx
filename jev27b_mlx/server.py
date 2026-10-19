"""POST /v1/decide. Body: {"state": ..., "question": {model question object}}  or the short form
{"state": ..., "kind": "choice"|"noul", "question": "text", "options": [...]}."""
from __future__ import annotations

import json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx

from .engine import Jev27


def serve(model: str, host: str = "127.0.0.1", port: int = 8000, memory_limit_gb: float | None = None):
    jev = Jev27(model, memory_limit_gb=memory_limit_gb)
    lock = threading.Lock()
    print(f"jev27b_mlx: loaded {model} in {jev.load_s:.1f}s ({mx.get_active_memory() / 1e9:.1f} GB), "
          f"prompt attention {jev.check['attention']}; POST http://{host}:{port}/v1/decide", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/v1/decide/info"):
                return self._send(200, {"model": model, "engine": "mlx", "kinds": ["choice", "noul"],
                                        "max_options": len(jev.codes), "prompt_attention": jev.check})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/decide":
                return self._send(404, {"error": {"message": "not found"}})
            try:
                r = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                q = r["question"]
                if not isinstance(q, dict):
                    q = Jev27.question(r.get("kind", "choice"), q, r.get("options"), r.get("descriptions"))
                t0 = time.perf_counter()
                with lock:
                    out = jev.decide(r.get("state"), q)
                out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                self._send(200, out)
            except (KeyError, ValueError, TypeError) as e:
                self._send(400, {"error": {"message": f"{type(e).__name__}: {e}"}})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})

        def log_message(self, *a):
            pass

    ThreadingHTTPServer((host, port), Handler).serve_forever()
