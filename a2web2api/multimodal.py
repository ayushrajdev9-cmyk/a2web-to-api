"""Gemini multimodal: Scotty resumable upload for image input.

Ported from gemini-web2api (MIT) and adapted to per-provider config.
"""

import re
import time
import urllib.request
import ssl

_ssl_ctx = None


def _get_ssl_ctx():
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


def _build_opener(proxy):
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
        urllib.request.HTTPSHandler(context=_get_ssl_ctx()),
    )


def detect_image_mime(image_bytes, fallback="image/png"):
    if not isinstance(image_bytes, bytes):
        return fallback
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(image_bytes) >= 12 and image_bytes[4:8] == b"ftyp":
        brand = image_bytes[8:12]
        if brand in (b"avif", b"avis"):
            return "image/avif"
        if brand in (b"heic", b"heix", b"hevc", b"hevx"):
            return "image/heic"
    return fallback


class GeminiUploader:
    """Uploads images to Gemini Web's Scotty resumable endpoint."""

    def __init__(self, cfg, cookie_fn):
        self.cfg = cfg
        self._cookie_fn = cookie_fn  # callable returning (cookie_str, extra)
        self._page_tokens_cache = {}
        self._tokens_ts = 0

    def _log(self, msg):
        from .util import log
        log(msg, enabled=self.cfg.get("log_requests", True), component="gemini")

    def _request(self, url, data, headers, timeout, proxy):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        if proxy:
            return _build_opener(proxy).open(req, timeout=timeout)
        return urllib.request.urlopen(req, context=_get_ssl_ctx(), timeout=timeout)

    def _get_page_tokens(self):
        now = time.time()
        if now - self._tokens_ts > 600:
            self._page_tokens_cache = self._fetch_page_tokens()
            self._tokens_ts = now
        return self._page_tokens_cache

    def _fetch_page_tokens(self):
        cookie_str, extra = self._cookie_fn()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        if cookie_str:
            headers["Cookie"] = cookie_str
        try:
            req = urllib.request.Request("https://gemini.google.com/app", headers=headers)
            proxy = self.cfg.get("proxy")
            if proxy:
                resp = _build_opener(proxy).open(req, timeout=30)
            else:
                resp = urllib.request.urlopen(req, context=_get_ssl_ctx(), timeout=30)
            html = resp.read().decode("utf-8", errors="replace")
            tokens = {}
            for key, pattern in [
                ("push_id", r'"qKIAYe":"([^"]+)"'),
                ("pctx", r'"Ylro7b":"([^"]+)"'),
                ("at", r'"thykhd":"([^"]+)"'),
            ]:
                m = re.search(pattern, html)
                if m:
                    tokens[key] = m.group(1)
            return tokens
        except Exception:
            return {}

    def upload(self, image_bytes, filename="image.png", mime_type="image/png"):
        tokens = self._get_page_tokens()
        push_id = tokens.get("push_id", "feeds/mcudyrk2a4khkz")
        pctx = tokens.get("pctx", "CgcSBWjK7pYx")
        cookie_str, extra = self._cookie_fn()
        from .providers.gemini import make_sapisidhash
        proxy = self.cfg.get("proxy")

        start_headers = {
            "Push-ID": push_id,
            "X-Tenant-Id": "bard-storage",
            "X-Client-Pctx": pctx,
            "X-Goog-Upload-Header-Content-Length": str(len(image_bytes)),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        if cookie_str:
            start_headers["Cookie"] = cookie_str
        if extra.get("sapisid"):
            start_headers["Authorization"] = make_sapisidhash(extra["sapisid"], "https://gemini.google.com")

        try:
            resp = self._request("https://content-push.googleapis.com/upload/", b"",
                                 start_headers, 30, proxy)
            upload_url = resp.headers.get("X-Goog-Upload-URL") or resp.headers.get("x-goog-upload-url")
        except Exception as e:
            raise RuntimeError(f"upload session start failed: {e}") from e
        if not upload_url:
            raise RuntimeError("no upload URL in response headers")

        upload_headers = {
            "X-Goog-Upload-Command": "upload, finalize",
            "X-Goog-Upload-Offset": "0",
            "Content-Type": "application/octet-stream",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        try:
            resp2 = self._request(upload_url, image_bytes, upload_headers, 60, proxy)
            file_ref = resp2.read().decode("utf-8", errors="replace").strip()
        except Exception as e:
            raise RuntimeError(f"upload finalize failed: {e}") from e
        if not file_ref or not file_ref.startswith("/"):
            raise RuntimeError(f"invalid file reference: {file_ref[:100]}")
        self._log(f"image uploaded: {filename} -> {file_ref[:60]}")
        return file_ref