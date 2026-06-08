"""Measurement proxy for L2. Every agent harness points here instead of at MLX
directly; the proxy forwards /v1/* to the real server and tallies tokens +
requests, so token-efficiency is measured uniformly regardless of how each
harness prompts internally. Control endpoints:
  POST /__bench/reset  -> zero the counters, returns {}
  GET  /__bench/stats  -> {requests, prompt_tokens, completion_tokens}

It injects stream_options.include_usage so streamed replies still carry usage.
Run ON tardis:  uv run python -m bench.proxy --upstream 11434 --port 11500
"""
from __future__ import annotations
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

LOCK = threading.Lock()
STATS = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
UPSTREAM = "http://127.0.0.1:11434"
FORCE_MODEL = ""   # if set, rewrite every request's model field to this exact id


def _tally(prompt, completion):
    with LOCK:
        STATS["requests"] += 1
        STATS["prompt_tokens"] += int(prompt or 0)
        STATS["completion_tokens"] += int(completion or 0)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path == "/__bench/stats":
            return self._json(200, dict(STATS))
        return self._forward("GET")

    def do_POST(self):
        if self.path == "/__bench/reset":
            with LOCK:
                STATS.update(requests=0, prompt_tokens=0, completion_tokens=0)
            return self._json(200, {})
        return self._forward("POST")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forward(self, method):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        streaming = False
        if raw:
            try:
                body = json.loads(raw)
                if body.get("stream"):
                    streaming = True
                    body.setdefault("stream_options", {})["include_usage"] = True
                if FORCE_MODEL and "model" in body:
                    body["model"] = FORCE_MODEL
                raw = json.dumps(body).encode()
            except json.JSONDecodeError:
                pass

        req = Request(UPSTREAM + self.path, data=raw if method == "POST" else None, method=method)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "connection"):
                req.add_header(k, v)
        try:
            up = urlopen(req, timeout=600)
        except Exception as e:  # noqa: BLE001
            return self._json(502, {"error": str(e)})

        self.send_response(up.status)
        ct = up.headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", ct)
        self.send_header("Connection", "close")
        self.end_headers()

        if streaming:
            self._pump_stream(up)
        else:
            data = up.read()
            try:
                usage = json.loads(data).get("usage") or {}
                _tally(usage.get("prompt_tokens"), usage.get("completion_tokens"))
            except json.JSONDecodeError:
                pass
            self.wfile.write(data)

    def _pump_stream(self, up):
        prompt = completion = 0
        for line in up:
            self.wfile.write(line)
            self.wfile.flush()
            s = line.decode("utf-8", "replace").strip()
            if s.startswith("data:"):
                payload = s[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        u = json.loads(payload).get("usage")
                        if u:
                            prompt = u.get("prompt_tokens", prompt)
                            completion = u.get("completion_tokens", completion)
                    except json.JSONDecodeError:
                        pass
        _tally(prompt, completion)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=int, default=11434)
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--model", default="", help="force every request's model id to this")
    args = ap.parse_args()
    global UPSTREAM, FORCE_MODEL
    UPSTREAM = f"http://127.0.0.1:{args.upstream}"
    FORCE_MODEL = args.model
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"proxy :{args.port} -> {UPSTREAM}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
