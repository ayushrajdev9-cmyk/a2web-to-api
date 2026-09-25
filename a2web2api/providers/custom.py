"""Custom config-driven provider — adapt ANY chat web service to this API.

No code required: define the endpoint, headers, request body template and
response paths in config.json under providers.custom.providers, e.g.:

    {
      "id": "perplexity",
      "label": "Perplexity",
      "models": {"perplexity-sonar": {"model": "sonar", "desc": "..."}},
      "endpoint": "https://www.perplexity.ai/rest/chat/answer",
      "headers": {"Content-Type": "application/json"},
      "body": {
        "query": "{transcript}",
        "model": "{model}",
        "attachments": {"$json": "attachments"}
      },
      "response": {
        "mode": "sse",
        "data_path": "answer",
        "final_text_path": "answer"
      }
    }

Body placeholders:
  {prompt}      last user message (plain text)
  {transcript}  full flattened conversation (plain text)
  {model}       upstream model id (plain text)
  {"$json": "messages"}      insert normalized message array (raw JSON)
  {"$json": "transcript"}    insert transcript as JSON string
  {"$json": "system"}        insert system text as JSON string
  {"$json": "prompt"}        insert last user text as JSON string

Response:
  mode: "sse" (default) or "json"
  event: optional SSE event name to filter on
  data_path: dotted path to text in each payload / json response
  final_text_path: dotted path for non-stream responses (falls back to data_path)

  Dotted paths support dict keys and list indices, e.g. "choices.0.delta.content".
"""

import json
import re

import httpx

from .base import BaseProvider
from ..cookie import load_cookie
from ..prompt import normalize_messages, transcript

PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")


def deep_get(obj, path):
    cur = obj
    for key in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list):
            try:
                cur = cur[int(key)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if cur is None:
            return None
    return cur


class CustomProvider(BaseProvider):
    name = "custom"
    label = "Custom (config-driven)"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.provider_id = cfg.get("id", "custom")
        self.cookie_file = cfg.get("cookie_file")
        self.proxy = cfg.get("proxy")
        self.timeout = cfg.get("request_timeout_sec", 180)
        self.retries = cfg.get("retry_attempts", 3)
        self.retry_delay = cfg.get("retry_delay_sec", 2)
        self.endpoint = cfg.get("endpoint")
        if not self.endpoint:
            raise ValueError(f"custom provider '{self.provider_id}': 'endpoint' is required")
        self.method = cfg.get("method", "POST").upper()
        self.headers = dict(cfg.get("headers") or {})
        self.body_template = cfg.get("body") or {}
        resp_cfg = cfg.get("response") or {}
        self.resp_mode = resp_cfg.get("mode", "sse")
        self.resp_event = resp_cfg.get("event")
        self.data_path = resp_cfg.get("data_path") or resp_cfg.get("final_text_path") or "choices.0.delta.content"
        self.final_text_path = resp_cfg.get("final_text_path") or self.data_path

        self.models = {}
        user_models = cfg.get("models") or {}
        for alias, meta in user_models.items():
            if isinstance(meta, str):
                meta = {"model": meta, "desc": alias}
            self.models[alias] = {"model": meta.get("model", alias), "desc": meta.get("desc", alias)}
        self.default_model = cfg.get("default_model") or (list(self.models)[0] if self.models else "")

    @property
    def name(self):
        return self.provider_id

    def status(self):
        cookie_str, _ = load_cookie(self.cookie_file)
        if cookie_str:
            return f"endpoint: {self.endpoint} | cookie: ok"
        return f"endpoint: {self.endpoint} | cookie: none"

    # ── template rendering ─────────────────────────────────────────────────

    def _render_value(self, value, context, messages):
        if isinstance(value, str):
            def repl(m):
                name = m.group(1)
                if name in context:
                    return str(context[name])
                return m.group(0)
            return PLACEHOLDER_RE.sub(repl, value)
        if isinstance(value, list):
            return [self._render_value(v, context, messages) for v in value]
        if isinstance(value, dict):
            if set(value.keys()) == {"$json"}:
                name = value["$json"]
                if name == "messages":
                    return normalize_messages(messages)
                if name == "transcript":
                    return transcript(messages)
                if name == "system":
                    sys_parts = [m["content"] for m in messages
                                 if isinstance(m, dict) and m.get("role") == "system"]
                    return "\n".join(sys_parts)
                if name == "prompt":
                    user_parts = [m["content"] for m in messages
                                  if isinstance(m, dict) and m.get("role") == "user"
                                  and isinstance(m.get("content"), str)]
                    return user_parts[-1] if user_parts else ""
                raise ValueError(f"custom provider: unknown $json source '{name}'")
            return {k: self._render_value(v, context, messages) for k, v in value.items()}
        return value

    def _last_user_text(self, messages):
        for m in reversed(messages or []):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str) and c.strip():
                    return c
                if isinstance(c, list):
                    parts = [p.get("text", "") for p in c
                             if isinstance(p, dict) and p.get("type") in ("text", "input_text")]
                    if "".join(parts).strip():
                        return "".join(parts)
        return ""

    # ── core ───────────────────────────────────────────────────────────────

    def _headers(self):
        cookie_str, _ = load_cookie(self.cookie_file)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/event-stream" if self.resp_mode == "sse" else "application/json",
        }
        for k, v in self.headers.items():
            if isinstance(v, str):
                headers[k] = v.replace("{cookie}", cookie_str)
            else:
                headers[k] = v
        if cookie_str and "Cookie" not in headers:
            headers["Cookie"] = cookie_str
        return headers

    def _body(self, messages, model):
        context = {
            "prompt": self._last_user_text(messages),
            "transcript": transcript(messages),
            "model": model,
        }
        return self._render_value(self.body_template, context, messages)

    def chat(self, messages, model, stream=False, images=None, tools=None, tool_choice=None, **kw):
        body = self._body(messages, model)
        if stream and self.resp_mode == "sse":
            return self._stream(body)
        if stream:
            raise ValueError("custom provider: 'stream' requires response.mode='sse'")
        return self._complete(body)

    def _request_args(self, body):
        args = {"headers": self._headers(), "timeout": self.timeout,
                "proxy": self.proxy, "follow_redirects": True}
        if self.method == "GET":
            args["params"] = body if isinstance(body, dict) else None
        else:
            args["json"] = body
        return args

    def _post_once(self, body):
        args = self._request_args(body)
        with httpx.Client(timeout=self.timeout, proxy=self.proxy, follow_redirects=True) as client:
            return client.request(self.method, self.endpoint, **args)

    def _extract(self, data, path):
        val = deep_get(data, path)
        return val if isinstance(val, str) else ""

    def _complete(self, body):
        last_err = None
        for attempt in range(self.retries):
            try:
                resp = self._post_once(body)
                if resp.status_code != 200:
                    raise RuntimeError(f"{self.provider_id} upstream {resp.status_code}: {resp.text[:300]}")
                if self.resp_mode == "json":
                    data = resp.json()
                    text = self._extract(data, self.final_text_path)
                    if not text:
                        raise RuntimeError(f"{self.provider_id} no text at '{self.final_text_path}': {str(data)[:300]}")
                    return text
                # SSE mode drained to completion
                from .deepseek import iter_sse
                final_text = ""
                for _, data_str in iter_sse(resp.iter_lines()):
                    try:
                        payload = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    text = self._extract(payload, self.data_path)
                    if text and len(text) > len(final_text):
                        final_text = text
                if not final_text:
                    raise RuntimeError(f"{self.provider_id} no text in SSE stream")
                return final_text
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    self.log(f"retry {attempt + 1}/{self.retries}: {e}")
                    import time
                    time.sleep(self.retry_delay)
        raise last_err

    def _stream(self, body):
        from .deepseek import iter_sse
        last_err = None
        for attempt in range(self.retries):
            try:
                with httpx.stream(
                    self.method, self.endpoint, json=body if self.method != "GET" else None,
                    params=body if self.method == "GET" else None,
                    headers=self._headers(), timeout=self.timeout,
                    proxy=self.proxy, follow_redirects=True,
                ) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"{self.provider_id} upstream {resp.status_code}")
                    emitted = ""
                    for event, data_str in iter_sse(resp.iter_lines()):
                        if self.resp_event and event != self.resp_event:
                            continue
                        if data_str == "[DONE]":
                            break
                        try:
                            payload = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        text = self._extract(payload, self.data_path)
                        if not text:
                            continue
                        if len(text) >= len(emitted) and text.startswith(emitted):
                            delta = text[len(emitted):]
                            emitted = text
                            if delta:
                                yield delta
                        else:
                            # cumulative changed shape; emit raw (rare)
                            yield text
                    if not emitted:
                        raise RuntimeError(f"{self.provider_id} stream ended with no text")
                return
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    self.log(f"stream retry {attempt + 1}/{self.retries}: {e}")
                    import time
                    time.sleep(self.retry_delay)
        raise last_err