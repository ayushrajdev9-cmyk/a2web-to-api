"""Gemini (gemini.google.com) provider.

Reverse-engineered StreamGenerate protocol, ported from gemini-web2api
(MIT) and generalized to per-provider configuration.

Why it works: the Gemini web app calls the internal
`assistant.lamda.BardFrontendService/StreamGenerate` endpoint with an
`f.req` protobuf-ish payload. We replay that request with the user's
cookies and parse the `"wrb.fr"` response lines for the model text.
"""

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import ssl
import uuid

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .base import BaseProvider
from ..cookie import load_cookie
from ..prompt import transcript
from ..util import estimate_tokens

# MODE_CATEGORY enum from Gemini frontend JS:
#   1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING, 6=FLASH_LITE

BUILTIN_MODELS = {
    "gemini-3.7-flash": {"mode": 1, "think": 4, "desc": "Latest all-around model (Gemini 3.7 Flash)"},
    "gemini-3.6-flash": {"mode": 1, "think": 4, "desc": "All-around model (Gemini 3.6 Flash)"},
    "gemini-3.5-flash": {"mode": 1, "think": 4, "desc": "Alias for gemini-3.6-flash"},
    "gemini-3.5-flash-thinking": {"mode": 2, "think": 0, "desc": "Deep thinking, longest output (~20k chars)"},
    "gemini-3.1-pro": {"mode": 3, "think": 4, "desc": "Pro model (needs cookie for real routing)"},
    "gemini-3.1-pro-enhanced": {"mode": 3, "think": 4, "extra": {31: 2, 80: 3}, "desc": "Pro with enhanced output"},
    "gemini-auto": {"mode": 4, "think": 4, "desc": "Auto model selection"},
    "gemini-3.5-flash-thinking-lite": {"mode": 5, "think": 0, "desc": "Dynamic thinking"},
    "gemini-flash-lite": {"mode": 6, "think": 4, "desc": "Lightweight fast model"},
}

_ssl_ctx = None


def _get_ssl_ctx():
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


def make_sapisidhash(sapisid: str, origin: str = "https://gemini.google.com") -> str:
    ts = int(time.time())
    h = hashlib.sha1(f"{ts} {sapisid} {origin}".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{h}"


class GeminiProvider(BaseProvider):
    name = "gemini"
    label = "Google Gemini"
    default_model = "gemini-3.6-flash"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.gemini_bl = cfg.get("gemini_bl", "boq_assistant-bard-web-server_20260716.08_p0")
        self.auth_user = cfg.get("auth_user")
        self.xsrf_token = cfg.get("xsrf_token")
        self.temporary_chats = cfg.get("temporary_chats", False)
        self.cookie_file = cfg.get("cookie_file")
        self.proxy = cfg.get("proxy")
        self.timeout = cfg.get("request_timeout_sec", 180)
        self.retries = cfg.get("retry_attempts", 3)
        self.retry_delay = cfg.get("retry_delay_sec", 2)

        # models: user config overrides builtin
        self.models = {}
        user_models = cfg.get("models") or {}
        for alias, meta in BUILTIN_MODELS.items():
            self.models[alias] = {"model": alias, "desc": meta["desc"]}
        for alias, meta in user_models.items():
            if isinstance(meta, str):
                meta = {"model": meta, "desc": alias}
            self.models[alias] = {"model": meta.get("model", alias), "desc": meta.get("desc", alias)}
        self._meta = {alias: (BUILTIN_MODELS[alias] if alias in BUILTIN_MODELS else None)
                      for alias in self.models}

        from ..multimodal import GeminiUploader
        self.uploader = GeminiUploader(cfg, self._cookie)

    # ── auth helpers ──────────────────────────────────────────────────────

    def _cookie(self):
        return load_cookie(self.cookie_file)

    def _account_prefix(self):
        if not self.auth_user:
            return ""
        return f"/u/{self.auth_user}"

    def _outgoing_headers(self):
        account_prefix = self._account_prefix()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://gemini.google.com",
            "Referer": f"https://gemini.google.com{account_prefix}/app",
            "X-Same-Domain": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        if account_prefix:
            headers["X-Goog-AuthUser"] = str(self.auth_user)
        cookie_str, extra = self._cookie()
        if cookie_str:
            headers["Cookie"] = cookie_str
        if extra.get("sapisid"):
            headers["Authorization"] = make_sapisidhash(extra["sapisid"])
        return headers

    def status(self):
        cookie_str, _ = self._cookie()
        if cookie_str:
            return "cookie: ok"
        return "cookie: none (anonymous mode, Pro routing unavailable)"

    # ── payload ────────────────────────────────────────────────────────────

    def _apply_persistence(self, inner):
        if self.temporary_chats:
            inner[41] = [1]
            inner[45] = 1
        else:
            inner[41] = [2]

    def _build_payload(self, prompt, model_id, think_mode, file_refs=None, extra_fields=None):
        inner = [None] * 102
        if file_refs:
            refs = [[None, None, ref] for ref in file_refs]
            inner[0] = [prompt, 0, None, refs, None, None, 0]
        else:
            inner[0] = [prompt, 0, None, None, None, None, 0]
        inner[1] = ["en"]
        inner[2] = ["", "", "", None, None, None, None, None, None, ""]
        inner[6] = [0]
        inner[7] = 1
        inner[10] = 1
        inner[11] = 0
        inner[17] = [[think_mode]]
        inner[18] = 0
        inner[27] = 1
        inner[30] = [4]
        self._apply_persistence(inner)
        inner[53] = 0
        inner[59] = str(uuid.uuid4())
        inner[61] = []
        inner[68] = 1
        inner[79] = model_id
        if extra_fields:
            for k, v in extra_fields.items():
                inner[k] = v
        outer = [None, json.dumps(inner)]
        params = {"f.req": json.dumps(outer)}
        if self.xsrf_token:
            params["at"] = self.xsrf_token
        return urllib.parse.urlencode(params)

    def _get_url(self):
        reqid = int(time.time()) % 1000000
        account_prefix = self._account_prefix()
        return (
            f"https://gemini.google.com{account_prefix}/_/BardChatUi/data/"
            "assistant.lamda.BardFrontendService/StreamGenerate"
            f"?bl={self.gemini_bl}&hl=en&_reqid={reqid}&rt=c"
        )

    # ── response parsing ───────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text, strip=True):
        text = re.sub(
            r'```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?',
            '', text, flags=re.DOTALL)
        text = re.sub(r'http://googleusercontent\.com/card_content/\d+\n?', '', text)
        return text.strip() if strip else text

    @staticmethod
    def _extract_texts_from_line(line):
        if '"wrb.fr"' not in line or len(line) < 200:
            return []
        try:
            arr = json.loads(line)
            inner_str = arr[0][2]
            if not inner_str or len(inner_str) < 50:
                return []
            inner = json.loads(inner_str)
            if not (isinstance(inner, list) and len(inner) > 4 and inner[4]):
                return []
            texts = []
            for part in inner[4]:
                if isinstance(part, list) and len(part) > 1 and part[1] and isinstance(part[1], list):
                    for t in part[1]:
                        if isinstance(t, str) and t:
                            texts.append(t)
            return texts
        except (json.JSONDecodeError, IndexError, TypeError):
            return []

    def extract_response_text(self, raw):
        bard_err = re.search(r'BardErrorInfo[^0-9]{0,8}\[?(\d+)\]?', raw)
        if bard_err:
            raise RuntimeError(f"Gemini rejected request: BardErrorInfo [{bard_err.group(1)}]")
        last_text = ""
        for line in raw.split("\n"):
            for t in self._extract_texts_from_line(line):
                if len(t) > len(last_text):
                    last_text = t
        return self._clean_text(last_text)

    # ── core ───────────────────────────────────────────────────────────────

    def _resolve_mode(self, model):
        """Map upstream model id to (mode_id, think_mode, extra_fields)."""
        if "@think=" in model:
            model, think_s = model.rsplit("@think=", 1)
            try:
                think = int(think_s)
            except ValueError:
                think = 4
        else:
            think = None
        meta = self._meta.get(model) or BUILTIN_MODELS.get(self.default_model, {})
        if not meta:
            meta = {"mode": 1, "think": 4}
        mode_id = meta.get("mode", 1)
        if think is None:
            think = meta.get("think", 4)
        return model, mode_id, think, meta.get("extra")

    def chat(self, messages, model, stream=False, tools=None, tool_choice=None, images=None, **kw):
        """Generate a reply; returns str or generator of str deltas."""
        prompt = transcript(messages, tools, tool_choice)
        if not prompt.strip():
            raise ValueError("empty prompt")
        model, mode_id, think_mode, extra_fields = self._resolve_mode(model)

        file_refs = None
        if images:
            refs = []
            for entry in images:
                resolved = None
                from ..prompt import resolve_image
                resolved = resolve_image(entry)
                if not resolved:
                    self.log("image resolve failed, skipping")
                    continue
                data, mime = resolved
                from ..multimodal import detect_image_mime
                mime = detect_image_mime(data, mime or "image/png")
                refs.append(self.uploader.upload(data, "image.png", mime))
            if refs:
                file_refs = refs

        if stream and HAS_HTTPX:
            return self._generate_stream(prompt, mode_id, think_mode, file_refs, extra_fields)
        if stream:
            # nobody has httpx: emulate by buffering
            def _fake_gen():
                yield self._generate(prompt, mode_id, think_mode, file_refs, extra_fields)
            return _fake_gen()
        return self._generate(prompt, mode_id, think_mode, file_refs, extra_fields)

    # ── transport ──────────────────────────────────────────────────────────

    def _generate(self, prompt, mode_id, think_mode, file_refs=None, extra_fields=None):
        body = self._build_payload(prompt, mode_id, think_mode, file_refs, extra_fields).encode()
        url = self._get_url()
        headers = self._outgoing_headers()
        last_err = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                if self.proxy:
                    opener = urllib.request.build_opener(
                        urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy}),
                        urllib.request.HTTPSHandler(context=_get_ssl_ctx()))
                    resp = opener.open(req, timeout=self.timeout)
                else:
                    resp = urllib.request.urlopen(req, context=_get_ssl_ctx(), timeout=self.timeout)
                raw = resp.read().decode("utf-8", errors="replace")
                text = self.extract_response_text(raw)
                if not text:
                    raise RuntimeError(
                        "Gemini returned an empty response (anonymous image upload requires "
                        "a cookie; configure providers.gemini.cookie_file)")
                return text
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    self.log(f"retry {attempt + 1}/{self.retries}: {e}")
                    time.sleep(self.retry_delay)
        raise last_err

    def _generate_stream(self, prompt, mode_id, think_mode, file_refs=None, extra_fields=None):
        body = self._build_payload(prompt, mode_id, think_mode, file_refs, extra_fields)
        url = self._get_url()
        headers = self._outgoing_headers()
        last_err = None
        emitted = ""
        for attempt in range(self.retries):
            try:
                with httpx.stream(
                    "POST", url, content=body, headers=headers,
                    timeout=self.timeout, proxy=self.proxy,
                    follow_redirects=True,
                ) as resp:
                    resp.raise_for_status()
                    buf = ""
                    for chunk in resp.iter_text():
                        buf += chunk
                        if "BardErrorInfo" in buf:
                            m = re.search(r'BardErrorInfo[^0-9]{0,8}\[?(\d+)\]?', buf)
                            if m:
                                raise RuntimeError(f"Gemini rejected request: BardErrorInfo [{m.group(1)}]")
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            for t in self._extract_texts_from_line(line):
                                if t == emitted:
                                    continue
                                if not t.startswith(emitted):
                                    raise RuntimeError("Gemini stream content changed")
                                delta = self._clean_text(t[len(emitted):], strip=False)
                                emitted = t
                                if delta:
                                    yield delta
                return
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    self.log(f"stream retry {attempt + 1}/{self.retries}: {e}")
                    time.sleep(self.retry_delay)
        raise last_err

    # ── capabilities ───────────────────────────────────────────────────────

    def supports_tools(self):
        return True

    def supports_images(self):
        return True