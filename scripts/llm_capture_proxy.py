"""Transparent logging proxy for capturing the exact LLM request SillyTavern
builds, for Rust-migration parity checks.

Sits between SillyTavern and llama.cpp: ST is pointed here for a single capture
request via `providerOptions.custom_url`, this logs the full request body (the
exact `messages`/prompt + params) and the response, then forwards verbatim to the
real llama.cpp server. The live game is unaffected — only the capture request
carries the custom_url override.

  python llm_capture_proxy.py [listen_port=8099] [upstream=http://127.0.0.1:8080]

Captures are written to ./captures/NNN-request.json and NNN-response.json next to
this script, and a one-line summary is printed per capture.
"""
import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
UPSTREAM = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080").rstrip("/")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(OUT_DIR, exist_ok=True)

_counter = {"n": 0}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Forward GETs (e.g. /v1/models for model resolution) verbatim, unlogged.
        try:
            with urllib.request.urlopen(f"{UPSTREAM}{self.path}", timeout=30) as resp:
                body, status = resp.read(), resp.status
                ctype = resp.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as err:
            body, status, ctype = err.read(), err.code, "application/json"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        _counter["n"] += 1
        n = _counter["n"]

        # Log the exact request body (the prompt ST built).
        try:
            parsed = json.loads(raw or b"{}")
            with open(os.path.join(OUT_DIR, f"{n:03d}-request.json"), "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            msgs = parsed.get("messages", [])
            sys_chars = len(msgs[0]["content"]) if msgs and msgs[0].get("role") == "system" else 0
            print(f"[capture {n:03d}] {self.path}  messages={len(msgs)}  system_chars={sys_chars}  "
                  f"stream={parsed.get('stream')}  model={parsed.get('model')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[capture {n:03d}] non-JSON body ({exc})", flush=True)

        # Forward verbatim to the real llama.cpp server.
        upstream_url = f"{UPSTREAM}{self.path}"
        req = urllib.request.Request(upstream_url, data=raw, method="POST")
        for key in ("Content-Type", "Authorization", "Accept"):
            if key in self.headers:
                req.add_header(key, self.headers[key])
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = resp.read()
                status = resp.status
                ctype = resp.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as err:
            body = err.read()
            status = err.code
            ctype = err.headers.get("Content-Type", "application/json")

        with open(os.path.join(OUT_DIR, f"{n:03d}-response.json"), "wb") as f:
            f.write(body)

        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


if __name__ == "__main__":
    print(f"llm_capture_proxy listening on 127.0.0.1:{PORT} -> {UPSTREAM}, writing to {OUT_DIR}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
