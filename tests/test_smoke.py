"""Smoke tests: config loading, provider construction, model routing, server boot.

No network calls. Run:  python -m pytest tests/  or  python tests/test_smoke.py
"""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2web2api.config import load_config
from a2web2api.providers import build_providers
from a2web2api.models import ModelResolver


def make_config():
    cfg = load_config()
    cfg["port"] = 0  # ephemeral
    cfg["providers"] = {
        "deepseek": {"enabled": True, "cookie_file": None},
        "chatgpt": {"enabled": True, "cookie_file": None},
        "grok": {"enabled": True, "cookie_file": None},
        "claude": {"enabled": True, "cookie_file": None},
        "gemini": {"enabled": True, "cookie_file": None},
    }
    # rebuild merge helpers after mutating providers dict
    from a2web2api.config import _merge_provider_defaults
    cfg["_provider"] = lambda name: _merge_provider_defaults(cfg, name)
    return cfg


def test_build_providers():
    cfg = make_config()
    providers = build_providers(cfg)
    assert "deepseek" in providers
    assert "gemini" in providers
    assert "grok" in providers
    assert "claude" in providers
    assert "chatgpt" in providers
    print("PASS build_providers:", ", ".join(providers))


def test_model_resolution():
    cfg = make_config()
    providers = build_providers(cfg)
    resolver = ModelResolver(providers, "deepseek-chat")
    pid, prov, upstream, meta, err = resolver.resolve("deepseek-chat")
    assert err is None and pid == "deepseek" and upstream == "deepseek-chat", (pid, err)
    pid, prov, upstream, meta, err = resolver.resolve("grok/grok-3")
    assert err is None and pid == "grok" and upstream == "grok-3", (pid, err)
    pid, prov, upstream, meta, err = resolver.resolve("chatgpt-4o")
    assert err is None and pid == "chatgpt" and upstream == "gpt-4o", (pid, err)
    pid, prov, upstream, meta, err = resolver.resolve("no/such-model")
    assert err is not None
    models = resolver.all_models()
    assert any(m["id"] == "deepseek-chat" for m in models)
    print("PASS model_resolution:", len(models), "models")


def test_chat_pipeline_offline():
    """Exercise the server's _handle_chat with a stub provider (no network)."""
    from a2web2api.server import A2WebHandler, ThreadingHTTPServerV2

    class StubProvider:
        name = "stub"
        label = "Stub"
        default_model = "stub-1"

        def __init__(self):
            self.cfg = {}

        def models_lookup(self):
            return {"stub-1": {"model": "stub-1", "desc": "stub"}}

        def status(self):
            return "stub"

        def supports_tools(self):
            return False

        def supports_images(self):
            return False

        def chat(self, messages, model, stream=False, **kw):
            text = "hello from stub"
            if stream:
                def gen():
                    for c in text:
                        yield c
                return gen()
            return text

        def close(self):
            pass

    class ResolverStub:
        def __init__(self):
            self.providers = {"stub": StubProvider()}

        def resolve(self, name):
            prov = self.providers["stub"]
            return "stub", prov, "stub-1", {"desc": "stub"}, None

        def all_models(self):
            return [{"id": "stub-1", "object": "model", "created": 0,
                     "owned_by": "stub", "description": "stub"}]

    server = ThreadingHTTPServerV2(("127.0.0.1", 0), A2WebHandler)
    server.providers = {"stub": StubProvider()}
    server.resolver = ResolverStub()
    server.api_keys = ["sk-test"]
    server.default_model = "stub-1"
    server.log_requests = False
    server.cfg = {}

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    port = server.server_address[1]

    import urllib.request
    body = json.dumps({
        "model": "stub-1",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }).encode()

    # unauthorized
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, timeout=10)
        assert False, "expected 401"
    except urllib.error.HTTPError as e:
        assert e.code == 401

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
    assert resp["choices"][0]["message"]["content"] == "hello from stub", resp
    assert resp["choices"][0]["finish_reason"] == "stop"

    # streaming
    body2 = json.dumps({"model": "stub-1", "messages": [{"role": "user", "content": "hi"}],
                        "stream": True}).encode()
    req2 = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body2,
        headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"})
    raw = urllib.request.urlopen(req2, timeout=10).read().decode()
    assert "data: [DONE]" in raw
    # reassemble streamed deltas (stub emits one char per chunk)
    parts = []
    for line in raw.split("\n"):
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        for ch in obj.get("choices", []):
            c = (ch.get("delta") or {}).get("content")
            if c:
                parts.append(c)
    assert "".join(parts) == "hello from stub", raw
    assert '"finish_reason": "stop"' in raw or '"finish_reason":"stop"' in raw, raw

    server.shutdown()
    print("PASS chat_pipeline_offline")


if __name__ == "__main__":
    test_build_providers()
    test_model_resolution()
    test_chat_pipeline_offline()
    print("ALL TESTS PASSED")