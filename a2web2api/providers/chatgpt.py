"""ChatGPT (chatgpt.com) provider.

The ChatGPT web app talks to:
    POST https://chatgpt.com/backend-api/conversation
with session cookies + an `oai-device-id` header. We replay that
endpoint with the OpenAI-compatible messages converted into the
webapp's `messages` format and diff-stream the answer text.

NOTE: ChatGPT's anti-bot stack (Arkose/Turnstile, Cloudflare) can
interrupt cookie-only access. If you see 403s, refresh cookies, avoid
datacenter IPs, or use residential proxies.
"""

import json
import uuid

import httpx

from .base import BaseProvider
from ..cookie import load_cookie
from ..prompt import normalize_messages
from .deepseek import iter_sse

BUILTIN_MODELS = {
    "chatgpt-4o": {"model": "gpt-4o", "desc": "ChatGPT 4o (web alias)"},
    "chatgpt-4o-mini": {"model": "gpt-4o-mini", "desc": "ChatGPT 4o mini"},
    "chatgpt-o3": {"model": "o3", "desc": "ChatGPT o3"},
    "chatgpt-o3-mini": {"model": "o3-mini", "desc": "ChatGPT o3 mini"},
    "chatgpt-o4-mini": {"model": "o4-mini", "desc": "ChatGPT o4 mini"},
}


class ChatGPTCookieProvider(BaseProvider):
    name = "chatgpt"
    label = "ChatGPT (web)"
    default_model = "chatgpt-4o"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cookie_file = cfg.get("cookie_file")
        self.proxy = cfg.get("proxy")
        self.timeout = cfg.get("request_timeout_sec", 180)
        self.retries = cfg.get("retry_attempts", 3)
        self.retry_delay = cfg.get("retry_delay_sec", 2)
        self.device_id = cfg.get("device_id") or str(uuid.uuid4())
        self.timezone_offset = cfg.get("timezone_offset_min", -120)

        self.models = {}
        user_models = cfg.get("models") or {}
        for alias, meta in BUILTIN_MODELS.items():
            self.models[alias] = {"model": meta["model"], "desc": meta["desc"]}
        for alias, meta in user_models.items():
            if isinstance(meta, str):
                meta = {"model": meta, "desc": alias}
            self.models[alias] = {"model": meta.get("model", alias), "desc": meta.get("desc", alias)}

    def status(self):
        cookie_str, _ = load_cookie(self.cookie_file)
        if cookie_str:
            return "cookie: ok (anti-bot may still 403)"
        return "cookie: none — REQUIRED (log into chatgpt.com and export cookies)"

    def _endpoint(self):
        return self.cfg.get("endpoint", "https://chatgpt.com/backend-api/conversation")

    def _headers(self):
        cookie_str, _ = load_cookie(self.cookie_file)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Origin": "https://chatgpt.com",
            "Referer": "https://chatgpt.com/",
            "Accept": "text/event-stream",
            "oai-device-id": self.device_id,
            "oai-language": "en-US",
        }
        if cookie_str:
            headers["Cookie"] = cookie_str
        extra = self.cfg.get("headers") or {}
        headers.update(extra)
        return headers

    def _to_webapp_messages(self, messages):
        """Convert OpenAI messages to the webapp messages format with UUIDs."""
        out = []
        for m in normalize_messages(messages, supported_roles=("system", "user", "assistant")):
            role, content = m["role"], m["content"]
            parts = [{"content_type": "text", "text": content}] if content else []
            if role == "assistant":
                out.append({
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": {"content_type": "text", "parts": [content]} if content else None,
                    "author": {"role": "assistant"},
                })
            elif role == "system":
                # openai webapp supports system via developer/instructions; fold into user
                out.append({
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": {"content_type": "text", "parts": [f"[System instructions] {content}"]},
                    "author": {"role": "user"},
                })
            else:
                out.append({
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": {"content_type": "text", "parts": [content]},
                    "author": {"role": "user"},
                })
        return out

    def _body(self, messages, model):
        webapp_msgs = self._to_webapp_messages(messages)
        body = {
            "action": "next",
            "messages": webapp_msgs,
            "parent_message_id": str(uuid.uuid4()),
            "model": model,
            "history_and_training_disabled": True,
            "conversation_mode": {"kind": "primary_assistant"},
            "timezone_offset_min": self.timezone_offset,
            "suggestions": [],
            "force_paragen": False,
            "force_rate_limit": False,
            "supported_encodings": ["text"],
        }
        overrides = self.cfg.get("body") or {}
        body.update(overrides)
        return body

    def chat(self, messages, model, stream=False, images=None, tools=None, tool_choice=None, **kw):
        if images:
            raise ValueError("ChatGPT web attachment upload is not implemented; text only")
        if tools and tool_choice != "none":
            raise ValueError("ChatGPT web tool calling is not implemented; text only")
        body = self._body(messages, model)
        if stream:
            return self._stream(body, model)
        return self._complete(body, model)

    def _post_once(self, body, model):
        return httpx.post(self._endpoint(), json=body, headers=self._headers(),
                          timeout=self.timeout, proxy=self.proxy, follow_redirects=True)

    @staticmethod
    def _extract_text(data_str):
        """Extract assistant text from a backend-api SSE payload."""
        if not data_str or data_str == "[DONE]":
            return ""
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return ""
        message = data.get("message")
        if not isinstance(message, dict):
            return ""
        author = message.get("author") or {}
        if author.get("role") != "assistant":
            return ""
        content = message.get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            return ""
        text = parts[0]
        return text if isinstance(text, str) else ""

    def _complete(self, body, model):
        last_err = None
        for attempt in range(self.retries):
            try:
                resp = self._post_once(body, model)
                if resp.status_code != 200:
                    raise RuntimeError(f"ChatGPT upstream {resp.status_code}: {resp.text[:300]}")
                final_text = ""
                for _, data_str in iter_sse(resp.iter_lines()):
                    text = self._extract_text(data_str)
                    if text and len(text) > len(final_text):
                        final_text = text
                if not final_text:
                    raise RuntimeError("ChatGPT returned no text")
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
                    "POST", self._endpoint(), json=body, headers=self._headers(),
                    timeout=self.timeout, proxy=self.proxy, follow_redirects=True,
                ) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"ChatGPT upstream {resp.status_code}")
                    emitted = ""
                    for _, data_str in iter_sse(resp.iter_lines()):
                        text = self._extract_text(data_str)
                        if text and len(text) > len(emitted) and text.startswith(emitted):
                            delta = text[len(emitted):]
                            emitted = text
                            if delta:
                                yield delta
                    if not emitted:
                        raise RuntimeError("ChatGPT stream ended with no text")
                return
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    self.log(f"stream retry {attempt + 1}/{self.retries}: {e}")
                    import time
                    time.sleep(self.retry_delay)
        raise last_err