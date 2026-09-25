"""HTTP server: OpenAI-compatible API endpoints (multi-provider).

Endpoints:
  GET  /                  status + provider overview
  GET  /health            liveness
  GET  /v1/models         model list
  POST /v1/chat/completions   OpenAI chat completions (stream + non-stream)
  POST /v1/responses          Responses API (Codex CLI style, best effort)
"""

import json
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .config import load_config, find_config
from .providers import build_providers
from .models import ModelResolver
from .prompt import extract_images, fence_pending, parse_tool_calls
from .util import (
    log, usage_from, new_completion_id, now,
    sse_chunk, sse_done, chunk_first, chunk_delta, chunk_end, chunk_tool_calls,
    non_stream_response,
)


class ThreadingHTTPServerV2(ThreadingHTTPServer):
    daemon_threads = True


class A2WebHandler(BaseHTTPRequestHandler):
    server_version = f"a2web2api/{__version__}"

    # ── HTTP plumbing ─────────────────────────────────────────────────────

    def log_message(self, fmt, *args):
        client = self.client_address[0] if self.client_address else "-"
        log(f"{client} {fmt % args}", enabled=self.server.log_requests, component="http")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _parse_body(self, body):
        if not body:
            return None
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None

    def _read_request_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _authorized(self):
        keys = self.server.api_keys
        if not keys:
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[7:] in keys:
            return True
        xk = self.headers.get("x-api-key") or self.headers.get("x-goog-api-key")
        if xk and xk in keys:
            return True
        if "?" in self.path:
            params = urllib.parse.parse_qs(self.path.split("?", 1)[1])
            for v in params.get("key", []):
                if v in keys:
                    return True
        return False

    def _require_v1_auth(self):
        if self.path.startswith("/v1") and not self._authorized():
            self.send_json({"error": {"message": "invalid api key"}}, 401)
            return False
        return True

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    # ── GET ───────────────────────────────────────────────────────────────

    def do_GET(self):
        try:
            if not self._require_v1_auth():
                return
            path = self.path.split("?", 1)[0]
            if path == "/":
                self.send_json({
                    "status": "ok",
                    "name": "a2web2api",
                    "version": __version__,
                    "providers": {
                        pid: {
                            "label": p.label,
                            "models": list(p.models_lookup().keys()),
                            "status": p.status(),
                        } for pid, p in self.server.providers.items()
                    },
                })
            elif path == "/health":
                self.send_json({"status": "ok", "version": __version__})
            elif path == "/v1/models":
                self.send_json({"object": "list", "data": self.server.resolver.all_models()})
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"GET error: {e}", enabled=self.server.log_requests)
            try:
                self.send_json({"error": {"message": str(e)}}, 500)
            except Exception:
                pass

    # ── POST ─────────────────────────────────────────────────────────────

    def do_POST(self):
        try:
            if not self._require_v1_auth():
                return
            path = self.path.split("?", 1)[0]
            body = self._read_request_body()
            if path == "/v1/chat/completions":
                self._handle_chat(body)
            elif path == "/v1/responses":
                self._handle_responses(body)
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"POST error: {e}", enabled=self.server.log_requests)
            try:
                self.send_json({"error": {"message": str(e)}}, 500)
            except Exception:
                pass

    # ── shared chat pipeline ─────────────────────────────────────────────

    def _prep(self, req):
        """Validate + resolve request to (pid, provider, upstream, messages, images,
        stream, tools, tool_choice). Raises ValueError for 400-class errors."""
        if req is None:
            raise ValueError("invalid JSON body")
        model_name = req.get("model") or self.server.default_model
        pid, prov, upstream, meta, err = self.server.resolver.resolve(model_name)
        if err:
            raise ValueError(err)
        messages = req.get("messages", [])
        if not messages:
            raise ValueError("messages required")
        stream = bool(req.get("stream", False))
        images = extract_images(messages)
        if images and not prov.supports_images():
            raise ValueError(f"provider '{pid}' does not support image input")
        tools = req.get("tools")
        tool_choice = req.get("tool_choice", "auto")
        # Providers without native tool support emulate them at the prompt
        # level (see prompt.tool_instruction), so tools are never rejected.
        return pid, prov, upstream, messages, images, stream, tools, tool_choice

    def _run(self, pid, prov, upstream, messages, images, stream, tools, tool_choice):
        """Execute provider; returns text (non-stream) or generator of deltas."""
        return prov.chat(messages, upstream, stream=stream, images=images,
                         tools=tools, tool_choice=tool_choice)

    def _handle_chat(self, body):
        req = self._parse_body(body)
        try:
            pid, prov, upstream, messages, images, stream, tools, tool_choice = self._prep(req)
        except ValueError as e:
            self.send_json({"error": {"message": str(e)}}, 400)
            return

        cid = new_completion_id()
        model_label = req.get("model") or self.server.default_model
        usage = usage_from("", "")
        text = ""

        if stream:
            try:
                gen = self._run(pid, prov, upstream, messages, images,
                                True, tools, tool_choice)
            except ValueError as e:
                self.send_json({"error": {"message": str(e)}}, 400)
                return
            except Exception as e:
                self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
                return

            try:
                self._start_sse()
                self.wfile.write(sse_chunk(chunk_first(cid, model_label)))
                self.wfile.flush()
                seen = ""
                emitted = 0
                for delta in gen:
                    if not delta:
                        continue
                    seen += delta
                    # Hold back anything that could still become a
                    # ```tool_call block; fence_pending needs the whole text so
                    # far to tell an opening fence from a closing one.
                    if not fence_pending(seen):
                        if seen[emitted:]:
                            self.wfile.write(
                                sse_chunk(chunk_delta(cid, model_label, seen[emitted:])))
                            self.wfile.flush()
                        emitted = len(seen)
                text = seen
                clean, tool_calls = parse_tool_calls(seen)
                if tool_calls:
                    # prose trapped in the held tail is still owed to the client;
                    # prose already streamed must not be sent twice
                    owed, _ = parse_tool_calls(seen[emitted:], strip=False)
                    self.wfile.write(sse_chunk(
                        chunk_tool_calls(cid, model_label, tool_calls, owed or None)))
                else:
                    if seen[emitted:]:
                        self.wfile.write(
                            sse_chunk(chunk_delta(cid, model_label, seen[emitted:])))
                        self.wfile.flush()
                    self.wfile.write(sse_chunk(chunk_end(cid, model_label, "stop")))
                self.wfile.write(sse_done())
                self.wfile.flush()
                log(f"[{pid}] streamed {len(seen)} chars", enabled=self.server.log_requests,
                    component="chat")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"[{pid}] stream error: {e}", enabled=self.server.log_requests,
                    component="chat")
                try:
                    self.wfile.write(sse_chunk(chunk_end(cid, model_label, "stop")))
                    self.wfile.write(sse_done())
                    self.wfile.flush()
                except Exception:
                    pass
            return

        # non-streaming
        try:
            text = self._run(pid, prov, upstream, messages, images, False, tools, tool_choice)
        except ValueError as e:
            self.send_json({"error": {"message": str(e)}}, 400)
            return
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        usage = usage_from("", text or "")
        msg = {"role": "assistant", "content": text or None}
        finish = "stop"
        if tools and tool_choice != "none" and text:
            clean, tool_calls = parse_tool_calls(text)
            if tool_calls:
                msg = {"role": "assistant", "content": clean or None, "tool_calls": tool_calls}
                finish = "tool_calls"
        if stream is False:
            self.send_json(non_stream_response(cid, model_label, msg, finish,
                                               usage, text or ""))
            return

    # ── /v1/responses (Codex CLI style) ──────────────────────────────────

    def _handle_responses(self, body):
        req = self._parse_body(body)
        try:
            if req is None:
                raise ValueError("invalid JSON body")
            messages = []
            instructions = req.get("instructions")
            if instructions:
                messages.append({"role": "system", "content": instructions})
            inp = req.get("input", [])
            if isinstance(inp, str):
                messages.append({"role": "user", "content": inp})
            elif isinstance(inp, list):
                for item in inp:
                    if isinstance(item, str):
                        messages.append({"role": "user", "content": item})
                    elif isinstance(item, dict):
                        t = item.get("type")
                        if t == "function_call_output":
                            messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                                             "name": item.get("name", ""),
                                             "content": item.get("output", "")})
                        elif t in ("input_text", "input_image", "image"):
                            messages.append({"role": "user", "content": [item]})
                        elif t == "message" and item.get("role") == "assistant":
                            acc = ""
                            for c in item.get("content", []) or []:
                                if isinstance(c, dict) and c.get("type") == "output_text":
                                    acc += c.get("text", "")
                            messages.append({"role": "assistant", "content": acc or None})
                        else:
                            messages.append({"role": item.get("role", "user"),
                                             "content": item.get("content", "")})
            if not messages:
                raise ValueError("no input provided")
            pid, prov, upstream, _, _, _, tools, _ = self._prep({**req, "messages": messages})
            text = self._run(pid, prov, upstream, messages, [],
                             False, tools, req.get("tool_choice", "auto"))
        except ValueError as e:
            self.send_json({"error": {"message": str(e)}}, 400)
            return
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        usage = usage_from("", text or "")
        self.send_json({
            "id": f"resp_{new_completion_id().split('-')[-1]}",
            "object": "response",
            "created_at": now(),
            "status": "completed",
            "model": req.get("model") or self.server.default_model,
            "output": [{
                "type": "message",
                "id": f"msg_{new_completion_id().split('-')[-1]}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text or "", "annotations": []}],
            }],
            "usage": {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        })


def serve(cfg: dict = None, config_path: str = None):
    """Build providers + resolver and run the threaded server (blocking)."""
    if cfg is None:
        cfg = load_config(config_path or find_config())

    providers = build_providers(cfg)
    if not providers:
        log("WARNING: no providers enabled — check config.json providers.*.enabled",
            component="a2web")

    server = ThreadingHTTPServerV2((cfg["host"], int(cfg["port"])), A2WebHandler)
    server.providers = providers
    server.resolver = ModelResolver(providers, cfg.get("default_model", ""))
    server.api_keys = cfg.get("api_keys") or []
    server.default_model = cfg.get("default_model", "deepseek-chat")
    server.log_requests = cfg.get("log_requests", True)
    server.cfg = cfg

    log(f"a2web2api v{__version__} — made by Ayush Rajdev & Anzar Iqbal", component="a2web")
    log(f"listening on http://{cfg['host']}:{cfg['port']}", component="a2web")
    log(f"auth: {'disabled' if not server.api_keys else 'api keys required (' + ', '.join(server.api_keys) + ')'}",
        component="a2web")
    for pid, prov in providers.items():
        log(f"  [provider] {pid:10s} {prov.label:22s} {prov.status()} | models: {len(prov.models_lookup())}",
            component="a2web")
    if not providers:
        log("  enable providers via config.json (providers.<name>.enabled=true)", component="a2web")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down", component="a2web")
        for prov in providers.values():
            try:
                prov.close()
            except Exception:
                pass
        server.server_close()

    return server
