"""Tests for cookie loading, validation, and provider auth guards.

No network calls. Run:  python -m pytest tests/  or  python tests/test_cookies.py

Made by Ayush Rajdev & Anzar Iqbal.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2web2api.cookie import cookie_status, load_cookie, looks_like_cookie
from a2web2api.providers.base import AuthRequired, BaseProvider


def test_looks_like_cookie_accepts_real_headers():
    for s in ("__Secure-ENID=abc; SID=xyz",
              "SID=xxx; HSID=yyy; SSID=zzz",
              "x-user-token=eyJhbGciOi.abc-_123",
              "d=1"):
        assert looks_like_cookie(s) is True, s


def test_looks_like_cookie_rejects_prose():
    """Notes left in a cookie file must not be sent as a session."""
    for s in ("PASTE YOUR COOKIE HERE\n\n1. Open site\n2. Press F12",
              "K=V; K2=V2\nRun this: cmd",
              "just some notes about the cookie",
              "", "   ", "=noName; ; ;"):
        assert looks_like_cookie(s) is False, s


def test_placeholder_file_is_not_accepted():
    """A file of instructions must read as no cookie, not as a valid session."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("PASTE YOUR DEEPSEEK COOKIE HERE\n\n"
                "1. Open https://chat.deepseek.com in Chrome and log in\n"
                "2. K=V; K2=V2\n")
        path = f.name
    try:
        cookie_str, _ = load_cookie(path)
        assert cookie_str == "", cookie_str
        assert cookie_status(path) == "stale", cookie_status(path)
    finally:
        os.unlink(path)


def test_real_cookie_is_loaded():
    raw = "__Secure-ENID=abc123; SID=xyz789; HSID=q1"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(raw)
        path = f.name
    try:
        cookie_str, _ = load_cookie(path)
        assert cookie_str == raw, cookie_str
        assert cookie_status(path) == "ok"
    finally:
        os.unlink(path)


def test_json_cookie_file_exposes_extras():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"cookie": "SID=abc", "sapisid": "SAPISID", "xsrf_token": "AOOh0P"},
                  f)
        path = f.name
    try:
        cookie_str, extra = load_cookie(path)
        assert cookie_str == "SID=abc", cookie_str
        assert extra.get("xsrf_token") == "AOOh0P", extra
    finally:
        os.unlink(path)


def test_netscape_cookie_file():
    body = ("# Netscape HTTP Cookie File\n"
            "chat.deepseek.com\tFALSE\t/\tTRUE\t0\tdsv1\tNESTED_VALUE\n"
            "#HttpOnly_chat.deepseek.com\tFALSE\t/\tTRUE\t0\ttoken\tABC123\n")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        cookie_str, _ = load_cookie(path)
        assert "token=ABC123" in cookie_str, cookie_str
    finally:
        os.unlink(path)


def test_missing_file_is_none():
    assert load_cookie("/nonexistent/cookie.txt") == ("", {})
    assert cookie_status("/nonexistent/cookie.txt") == "none"
    assert cookie_status(None) == "none"


def test_require_cookie_raises_actionable_error():
    """A cookie-gated provider must fail with instructions, not a parse error."""
    from a2web2api.providers.deepseek import DeepSeekProvider

    prov = DeepSeekProvider({"cookie_file": "/nonexistent/cookie.txt"})
    try:
        prov.chat([{"role": "user", "content": "hi"}], "deepseek-chat")
        assert False, "expected AuthRequired"
    except AuthRequired as e:
        msg = str(e)
        assert "deepseek" in msg
        assert "cookie_file" in msg
        assert "config.json" in msg
    # AuthRequired must reach the client as a 400, not an opaque 502
    assert issubclass(AuthRequired, ValueError)


def test_auth_required_is_value_error():
    assert issubclass(AuthRequired, ValueError)
    assert hasattr(BaseProvider, "require_cookie")


if __name__ == "__main__":
    test_looks_like_cookie_accepts_real_headers()
    test_looks_like_cookie_rejects_prose()
    test_placeholder_file_is_not_accepted()
    test_real_cookie_is_loaded()
    test_json_cookie_file_exposes_extras()
    test_netscape_cookie_file()
    test_missing_file_is_none()
    test_require_cookie_raises_actionable_error()
    test_auth_required_is_value_error()
    print("ALL COOKIE TESTS PASSED")
