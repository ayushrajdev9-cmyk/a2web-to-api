"""DeepSeek (chat.deepseek.com) provider.

The DeepSeek web app talks to a native OpenAI-compatible endpoint:
    POST https://chat.deepseek.com/api/v0/chat/completions
with the user's session cookies. Body is OpenAI-format messages +
`stream` + `model` (deepseek-chat / deepseek-reasoner). Response SSE
is OpenAI-format chunks, which we can forward almost verbatim.
"""

import json
import time

import httpx

from .base import BaseProvider
from ..cookie import load_cookie
from ..prompt import fold_tool_messages, normalize_messages, with_tool_instruction

BUILTIN_MODELS = {
    "deepseek-chat": {"desc": "DeepSeek V3 general chat model"},
    "deepseek-reasoner": {"desc": "DeepSeek R1 reasoning model"},
    "deepseek-coder": {"desc": "DeepSeek coding model (web alias)"},
}


class DeepSeekProvider(BaseProvider):
    name = "deepseek"
    label = "DeepSeek"
    default_model = "deepseek-chat"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cookie_file = cfg.get("cookie_file")
        self.proxy = cfg.get("proxy")
        self.timeout = cfg.get("request_timeout_sec", 180)
        self.retries = cfg.get("retry_attempts", 3)
        self.retry_delay = cfg.get("retry_delay_sec", 2)
        self.x_app_version = cfg.get("x_app_version", "20241129.1")

        self.models = {}
        user_models = cfg.get("models") or {}
        for alias, meta in BUILTIN_MODELS.items():
            self.models[alias] = {"model": alias, "desc": meta["desc"]}
        for alias, meta in user_models.items():
            if isinstance(meta, str):
                meta = {"model": meta, "desc": alias}
            self.models[alias] = {"model": meta.get("model", alias), "desc": meta.get("desc", alias)}

    def _endpoint(self):
        return self.cfg.get("endpoint", "https://chat.deepseek.com/api/v0/chat/completions")

    def status(self):
        cookie_str, _ = load_cookie(self.cookie_file)
        if cookie_str:
            return "cookie: ok"
        return "cookie: none — REQUIRED (log into chat.deepseek.com and export cookies)"

    # ── core ───────────────────────────────────────────────────────────────

    def chat(self, messages, model, stream=False, images=None, tools=None,
             tool_choice=None, **kw):
        self.require_cookie()
        if images:
            raise ValueError("DeepSeek web does not support image input")
        msgs = normalize_messages(
            fold_tool_messages(with_tool_instruction(messages, tools, tool_choice or "auto")),
            supported_roles=("system", "user", "assistant"))
        if not any(m["content"].strip() for m in msgs if m["role"] != "system") and \
           not any(m["content"].strip() for m in msgs if m["role"] == "user"):
            raise ValueError("empty prompt")

        body = {
            "messages": msgs,
            "model": model,
            "stream": stream,
        }
        if self.cfg.get("chat_template_kwargs"):
            body.update(self.cfg["chat_template_kwargs"])

        if stream:
            return self._stream(body)
        return self._complete(body)

    def _headers(self):
        cookie_str, _ = load_cookie(self.cookie_file)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Origin": "https://chat.deepseek.com",
            "Referer": "https://chat.deepseek.com/",
            "x-app-version": self.x_app_version,
            "Accept": "text/event-stream",
        }
        if cookie_str:
            headers["Cookie"] = cookie_str
        return headers

    def _post(self, body):
        last_err = None
        for attempt in range(self.retries):
            try:
                with httpx.Client(timeout=self.timeout, proxy=self.proxy, follow_redirects=True) as client:
                    return client.post(self._endpoint(), json=body, headers=self._headers())
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    self.log(f"retry {attempt + 1}/{self.retries}: {e}")
                    time.sleep(self.retry_delay)
        raise last_err

    def _complete(self, body):
        resp = self._post(body)
        if resp.status_code != 200:
            raise RuntimeError(f"DeepSeek upstream {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected DeepSeek response: {str(data)[:300]}") from e

    def _stream(self, body):
        last_err = None
        for attempt in range(self.retries):
            try:
                with httpx.stream(
                    "POST", self._endpoint(), json=body, headers=self._headers(),
                    timeout=self.timeout, proxy=self.proxy, follow_redirects=True,
                ) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"DeepSeek upstream {resp.status_code}")
                    emitted = ""
                    for event, data_str in iter_sse(resp.iter_lines()):
                        if data_str == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                        text = delta.get("content")
                        if text:
                            yield text
                            emitted += text
                return
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    self.log(f"stream retry {attempt + 1}/{self.retries}: {e}")
                    time.sleep(self.retry_delay)
        raise last_err


def iter_sse(lines):
    """Yield (event, data_str) from iterable of raw lines."""
    event = None
    data_parts = []
    for line in lines:
        line = (line or "").rstrip("\r\n")
        if line == "":
            if data_parts:
                yield event, "\n".join(data_parts)
            event, data_parts = None, []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
    if data_parts:
        yield event, "\n".join(data_parts)