"""Shared cookie file loader with mtime-based caching.

Accepted formats (same as gemini-web2api):
  * plain text:  "SID=xxx; HSID=xxx; SSID=xxx; ..."
  * JSON:        {"cookie": "SID=xxx; ...", "sapisid": "xxx", "xsrf_token": "..."}
  * Netscape:    exported cookie.txt lines (header skipped, K=V pairs joined)
"""

import json
import os
import time

_cache = {"path": None, "mtime": 0, "cookie_str": "", "extra": {}}


def load_cookie(cookie_file: str, proxy: str = None):
    """Load cookie for a provider.

    Returns (cookie_str, extra_dict). extra_dict holds provider-specific
    fields found in JSON cookie files (e.g. sapisid, xsrf_token, org_id).
    """
    if not cookie_file or not os.path.exists(cookie_file):
        return "", {}
    if cookie_file == _cache["path"] and os.path.getmtime(cookie_file) == _cache["mtime"]:
        return _cache["cookie_str"], _cache["extra"]

    cookie_str, extra = "", {}
    try:
        with open(cookie_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("{"):
            data = json.loads(content)
            cookie_str = data.get("cookie", "")
            extra = {k: v for k, v in data.items() if k != "cookie"}
        elif "\t" in content and any(line.startswith(("#", "")) and "\t" in line for line in content.splitlines()):
            # Netscape cookie file
            pairs = []
            for line in content.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 7:
                    pairs.append(f"{parts[5]}={parts[6]}")
            cookie_str = "; ".join(pairs)
        else:
            cookie_str = content
            pairs = dict(p.split("=", 1) for p in cookie_str.split("; ") if "=" in p)
            if "SAPISID" in pairs:
                extra["sapisid"] = pairs["SAPISID"]
    except Exception:
        pass

    _cache.update({"path": cookie_file, "mtime": os.path.getmtime(cookie_file),
                   "cookie_str": cookie_str, "extra": extra})
    return cookie_str, extra


def netscape_to_json(src: str, dst: str) -> None:
    """Convert a Netscape cookie file to a JSON cookie file."""
    cookie_str, _ = load_cookie(src)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"cookie": cookie_str}, f, indent=2)


def cookie_status(cookie_file: str) -> str:
    """Human-readable status: 'none' | 'ok' | 'stale' (file exists but empty)."""
    if not cookie_file or not os.path.exists(cookie_file):
        return "none"
    try:
        mtime = os.path.getmtime(cookie_file)
        if time.time() - mtime > 7 * 86400:
            return "stale"
    except OSError:
        pass
    cookie_str, _ = load_cookie(cookie_file)
    return "ok" if cookie_str else "stale"