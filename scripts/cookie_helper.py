#!/usr/bin/env python3
"""Cookie helper for a2web2api.

Export cookies for a site in the `K=V; K2=V2` single-line format (or
Netscape format) and this tool converts them to the JSON cookie file
that a2web2api reads.

Usage:
  python scripts/cookie_helper.py convert netscape_cookies.txt cookies/grok.txt
  python scripts/cookie_helper.py check cookies/grok.txt
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


def cmd_dump_json(cookie_str, dst):
    """Build a JSON cookie file from a pasted single-line cookie string."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"cookie": cookie_str}, f, indent=2)
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "convert" and len(sys.argv) == 4:
        cmd_convert(sys.argv[2], sys.argv[3])
    elif cmd == "check" and len(sys.argv) == 3:
        cmd_check(sys.argv[2])
    elif cmd == "from-string" and len(sys.argv) == 4 and sys.stdin.isatty():
        cmd_dump_json(sys.argv[2], sys.argv[3])
    else:
        print("usage: python scripts/cookie_helper.py {convert|check} ...")
        print(__doc__)
        sys.exit(1)