"""Grok (grok.com) provider.

The grok.com web app creates conversations through a private REST API:
    POST https://grok.com/rest/app-chat/conversations/new
authored with the user's session cookies. Generation streams back as
SSE events named `conversation`; the answer text accumulates in
`data.result.message` (and mirrors in `result.response`).

The parser below is intentionally tolerant — it hunts for whichever
text-bearing field grok.com is currently using and emits deltas.
"""

import json

import httpx

from .base import BaseProvider
from ..cookie import load_cookie
from ..prompt import transcript
from .deepseek import iter_sse

BUILTIN_MODELS = {
    "grok-4": {"desc": "Grok 4 (latest flagship)"},
    "grok-3": {"desc": "Grok 3"},
    "grok-3-fast": {"desc": "Grok 3 fast / mini variant"},
    "grok-3-thinking": {"desc": "Grok 3 with reasoning enabled"},
    "grok-2": {"desc": "Grok 2"},
}


def _deep_get(obj, dotted, default=None):
    cur = obj
    for key in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list):
            try:
                cur = cur[int(key)]
            except (ValueError, IndexError):
                return default
        else:
            return default
        if cur is None:
            return default
    return cur


class GrokProvider(BaseProvider):
    name = "grok"
    label = "xAI Grok"
    default_model = "grok-4"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cookie_file = cfg.get("cookie_file")
        self.proxy = cfg.get("proxy")
        self.timeout = cfg.get("request_timeout_sec", 180)
        self.retries = cfg.get("retry_attempts", 3)
        self.retry_delay = cfg.get("retry_delay_sec", 2)
        self.is_reasoning = cfg.get("is_reasoning", False)

        self.models = {}
        user_models = cfg.get("models") or {}
        for alias, meta in BUILTIN_MODELS.items():
            self.models[alias] = {"model": alias, "desc": meta["desc"]}
        for alias, meta in user_models.items():
            if isinstance(meta, str):
                meta = {"model": meta, "desc": alias}
            self.models[alias] = {"model": meta.get("model", alias), "desc": meta.get("desc", alias)}

    def _endpoint(self, model):
        return self.cfg.get("endpoint", "https://grok.com/rest/app-chat/conversations/new")

    def status(self):
        cookie_str, _ = load_cookie(self.cookie_file)
        if cookie_str:
            return "cookie: ok"
        return "cookie: none — REQUIRED (log into grok.com and export cookies)"

    def _headers(self):
        cookie_str, _ = load_cookie(self.cookie_file)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Origin": "https://grok.com",
            "Referer": "https://grok.com/",
            "Accept": "text/event-stream",
        }
        if cookie_str:
            headers["Cookie"] = cookie_str
        extra = self.cfg.get("headers") or {}
        headers.update(extra)
        return headers

    def _body(self, prompt, model):
        body = {
            "temporary": True,
            "modelName": model,
            "message": prompt,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "customInstructions": None,
            "returnImageBytes": False,
            "enableImageGen": False,
            "imageGenStyle": "",
            "returnRawGrokInXaiRequest": False,
            "deepsearchPreset": None,
            "isReasoning": self.is_reasoning,
        }
        overrides = self.cfg.get("body") or {}
        body.update(overrides)
        return body

    def chat(self, messages, model, stream=False, images=None, tool_choice=None, tools=None, **kw):
        self.require_cookie()
        if images:
            raise ValueError("Grok web image upload is not implemented; text only")
        prompt = transcript(messages, tools, tool_choice)
        if not prompt.strip():
            raise ValueError("empty prompt")
        body = self._body(prompt, model)
        if stream:
            return self._stream(body, model)
        return self._complete(body, model)

    def _post_once(self, body):
        headers = self._headers()
        return httpx.post(self._endpoint(body.get("modelName", "")), json=body, headers=headers,
                          timeout=self.timeout, proxy=self.proxy, follow_redirects=True)

    def _complete(self, body, model):
        last_err = None
        for attempt in range(self.retries):
            try:
                resp = self._post_once(body)
                if resp.status_code != 200:
                    raise RuntimeError(f"Grok upstream {resp.status_code}: {resp.text[:300]}")
                # non-stream fallback: drain SSE and take the longest text seen
                final_text = ""
                for _, data_str in iter_sse(resp.iter_lines()):
                    text = self._extract_text(data_str)
                    if text and len(text) > len(final_text):
                        final_text = text
                if not final_text:
                    raise RuntimeError("Grok returned no text")
                return final_text
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
                    "POST", self._endpoint(model), json=body, headers=self._headers(),
                    timeout=self.timeout, proxy=self.proxy, follow_redirects=True,
                ) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"Grok upstream {resp.status_code}")
                    emitted = ""
                    for _, data_str in iter_sse(resp.iter_lines()):
                        text = self._extract_text(data_str)
                        if text and len(text) > len(emitted) and text.startswith(emitted):
                            delta = text[len(emitted):]
                            emitted = text
                            if delta:
                                yield delta
                    if not emitted:
                        raise RuntimeError("Grok stream ended with no text")
                return
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    self.log(f"stream retry {attempt + 1}/{self.retries}: {e}")
                    import time
                    time.sleep(self.retry_delay)
        raise last_err

    def _extract_text(self, data_str):
        """Find the accumulated answer text in a grok SSE payload."""
        if not data_str or data_str == "[DONE]":
            return ""
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return ""
        result = _deep_get(data, "data.result")
        if not isinstance(result, dict):
            result = data if isinstance(data, dict) else None
        if not isinstance(result, dict):
            return ""
        for path in ("result.message", "result.response", "message", "response"):
            pass
        candidates = []
        # walk result for string fields commonly used for text
        for key in ("message", "response", "text", "content"):
            val = result.get(key)
            if isinstance(val, str) and val:
                candidates.append(val)
        rings = result.get("rings")
        if isinstance(rings, dict) and rings.get("answer"):
            candidates.append(rings["answer"])
        if not candidates:
            return ""
        return max(candidates, key=len)