#!/usr/bin/env python3
"""Cookie helper for a2web2api.

Export cookies for a site in the `K=V; K2=V2` single-line format (or
Netscape format) and this tool converts them to the JSON cookie file
that a2web2api reads.

Usage:
  python scripts/cookie_helper.py convert netscape_cookies.txt cookies/grok.txt
  python scripts/cookie_helper.py check cookies/grok.txt
  python scripts/cookie_helper.py set deepseek "K=V; K2=V2"
  echo "K=V; K2=V2" | python scripts/cookie_helper.py set deepseek
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2web2api.cookie import load_cookie, netscape_to_json, cookie_status


def cmd_convert(src, dst):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    netscape_to_json(src, dst)
    s, _ = load_cookie(dst)
    print(f"converted {src} -> {dst} ({len(s)} cookie bytes)")
    print("cookie status:", cookie_status(dst))


def cmd_check(path):
    s, extra = load_cookie(path)
    print(f"file      : {path}")
    print(f"status    : {cookie_status(path)}")
    print(f"length    : {len(s)} chars")
    if extra:
        print(f"extra keys: {', '.join(extra)}")


def cmd_set(provider, cookie_str, root):
    """Write a pasted cookie string to cookies/<provider>.txt and point config at it."""
    cookie_str = (cookie_str or "").strip()
    if not cookie_str:
        sys.exit("no cookie given: pass it as an argument or pipe it on stdin")
    if "=" not in cookie_str:
        sys.exit("that does not look like a cookie (expected 'K=V; K2=V2')")

    dst = os.path.join(root, "cookies", f"{provider}.txt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"cookie": cookie_str}, f, indent=2)
    os.chmod(dst, 0o600)
    s, _ = load_cookie(dst)
    print(f"wrote {dst} ({len(s)} cookie bytes, mode 600)")

    # point config.json at it, if a config exists
    cfg_path = os.path.join(root, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        block = cfg.setdefault("providers", {}).setdefault(provider, {})
        block["enabled"] = True
        rel = os.path.relpath(dst, root)
        block["cookie_file"] = rel
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"config.json: providers.{provider}.cookie_file = {rel!r}")
    print("restart the server to pick it up: ./run.sh")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "convert" and len(sys.argv) == 4:
        cmd_convert(sys.argv[2], sys.argv[3])
    elif cmd == "check" and len(sys.argv) == 3:
        cmd_check(sys.argv[2])
    elif cmd == "set" and len(sys.argv) >= 3:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw = sys.argv[3] if len(sys.argv) > 3 else sys.stdin.read()
        cmd_set(sys.argv[2], raw, root)
    else:
        print(__doc__)
        sys.exit(1)
        sys.exit(1)