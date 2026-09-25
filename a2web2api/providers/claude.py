"""Claude (claude.ai) provider.

The claude.ai web app talks to:
    POST https://claude.ai/api/chat
with session cookies and (optionally) an organization UUID. Streams
back Anthropic-style SSE (`content_block_delta` / `text_delta`).

Multi-turn is flattened into a single text block per turn — the web
endpoint is one-shot per request, matching the gemini-web2api model.
"""

import json

import httpx

from .base import BaseProvider
from ..cookie import load_cookie
from ..prompt import transcript
from .deepseek import iter_sse

BUILTIN_MODELS = {
    "claude-opus-4": {"desc": "Claude Opus 4 (web alias)"},
    "claude-sonnet-4": {"desc": "Claude Sonnet 4 (web alias)"},
    "claude-3-7-sonnet": {"desc": "Claude 3.7 Sonnet"},
    "claude-3-5-haiku": {"desc": "Claude 3.5 Haiku (web alias)"},
    "claude-2.1": {"desc": "Claude 2.1 (legacy)"},
}


class ClaudeProvider(BaseProvider):
    name = "claude"
    label = "Anthropic Claude"
    default_model = "claude-sonnet-4"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cookie_file = cfg.get("cookie_file")
        self.proxy = cfg.get("proxy")
        self.timeout = cfg.get("request_timeout_sec", 180)
        self.retries = cfg.get("retry_attempts", 3)
        self.retry_delay = cfg.get("retry_delay_sec", 2)
        self.organization = cfg.get("organization")  # optional UUID; auto-discovered otherwise
        self._org_cache = {"uuid": None, "ts": 0}

        self.models = {}
        user_models = cfg.get("models") or {}
        for alias, meta in BUILTIN_MODELS.items():
            self.models[alias] = {"model": alias, "desc": meta["desc"]}
        for alias, meta in user_models.items():
            if isinstance(meta, str):
                meta = {"model": meta, "desc": alias}
            self.models[alias] = {"model": meta.get("model", alias), "desc": meta.get("desc", alias)}

    def status(self):
        cookie_str, _ = load_cookie(self.cookie_file)
        if cookie_str:
            return "cookie: ok"
        return "cookie: none — REQUIRED (log into claude.ai and export cookies)"

    def _headers(self, with_org=True):
        cookie_str, _ = load_cookie(self.cookie_file)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Origin": "https://claude.ai",
            "Referer": "https://claude.ai/",
            "Accept": "text/event-stream",
            "anthropic-client": "web",
        }
        if cookie_str:
            headers["Cookie"] = cookie_str
        if with_org:
            org = self._get_org()
            if org:
                headers["anthropic-organization"] = org
        extra = self.cfg.get("headers") or {}
        headers.update(extra)
        return headers

    def _discover_org(self):
        import time
        try:
            with httpx.Client(timeout=30, proxy=self.proxy, follow_redirects=True) as client:
                resp = client.get("https://claude.ai/api/organizations", headers=self._headers(with_org=False))
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        return data[0].get("uuid")
        except Exception:
            pass
        return None

    def _get_org(self):
        import time
        now = time.time()
        if self.organization:
            return self.organization
        if now - self._org_cache["ts"] > 3600:
            self._org_cache["uuid"] = self._discover_org()
            self._org_cache["ts"] = now
        return self._org_cache["uuid"]

    def _endpoint(self):
        return self.cfg.get("endpoint", "https://claude.ai/api/chat")

    def _body(self, prompt, model):
        body = {
            "conversation_id": None,
            "text": prompt,
            "model": model,
            "attachments": [],
        }
        org = self._get_org()
        if org:
            body["organization"] = org
        overrides = self.cfg.get("body") or {}
        body.update(overrides)
        return body

    def chat(self, messages, model, stream=False, images=None, tools=None, tool_choice=None, **kw):
        if images:
            # base64 attachment would require claude.ai attachment upload flow
            raise ValueError("Claude web attachment upload is not implemented; text only")
        if tools and tool_choice != "none":
            raise ValueError("Claude web tool calling is not implemented; text only")
        prompt = transcript(messages)
        if not prompt.strip():
            raise ValueError("empty prompt")
        body = self._body(prompt, model)
        if stream:
            return self._stream(body, model)
        return self._complete(body, model)

    def _post_once(self, body, model):
        return httpx.post(self._endpoint(), json=body, headers=self._headers(),
                          timeout=self.timeout, proxy=self.proxy, follow_redirects=True)

    def _complete(self, body, model):
        last_err = None
        for attempt in range(self.retries):
            try:
                resp = self._post_once(body, model)
                if resp.status_code != 200:
                    raise RuntimeError(f"Claude upstream {resp.status_code}: {resp.text[:300]}")
                text_parts = []
                for _, data_str in iter_sse(resp.iter_lines()):
                    delta = self._extract_delta(data_str)
                    if delta:
                        text_parts.append(delta)
                text = "".join(text_parts)
                if not text:
                    raise RuntimeError("Claude returned no text")
                return text
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    self.log(f"retry {attempt + 1}/{self.retries}: {e}")
                    import time
                    time.sleep(self.retry_delay)
        raise last_err

    def _stream(self, body, model):
        last_err = None
        for attempt in range(self.retries):
            try:
                with httpx.stream(
                    "POST", self._endpoint(), json=body, headers=self._headers(),
                    timeout=self.timeout, proxy=self.proxy, follow_redirects=True,
                ) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"Claude upstream {resp.status_code}")
                    emitted = 0
                    for _, data_str in iter_sse(resp.iter_lines()):
                        delta = self._extract_delta(data_str)
                        if delta:
                            yield delta
                            emitted += len(delta)
                    if not emitted:
                        raise RuntimeError("Claude stream ended with no text")
                return
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    self.log(f"stream retry {attempt + 1}/{self.retries}: {e}")
                    import time
                    time.sleep(self.retry_delay)
        raise last_err

    @staticmethod
    def _extract_delta(data_str):
        if not data_str or data_str == "[DONE]":
            return ""
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return ""
        if data.get("type") == "content_block_delta":
            delta = data.get("delta") or {}
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                return delta.get("text", "")
        return ""