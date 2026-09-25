"""OpenAI message normalization and tool-call parsing.

Any provider that works from a flat text prompt (Gemini, Grok, Claude,
custom transcript templates) consumes the output of `transcript()`.
Providers with native message-array support (DeepSeek, ChatGPT, custom)
use `normalize_messages()` instead.
"""

import base64
import binascii
import json
import re
import uuid
from urllib.parse import unquote_to_bytes

MAX_IMAGE_B64_SIZE = 50000  # web-prompt size cap for inline base64 images


def normalize_messages(messages, supported_roles=("system", "user", "assistant")):
    """Return a list of {"role", "content"} with list-content flattened.

    - content parts of type text/input_text are joined.
    - image parts are replaced with "[Image attached]" tags unless
      extract_images is run first.
    """
    out = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        if role not in supported_roles:
            role = "user"
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                t = c.get("type", "")
                if t in ("text", "input_text", "output_text"):
                    parts.append(c.get("text", ""))
                elif t in ("image_url", "image", "input_image"):
                    parts.append("[Image attached]")
            content = " ".join(p for p in parts if p)
        if isinstance(content, (int, float)):
            content = str(content)
        out.append({"role": role, "content": str(content)})
    return out


def extract_images(messages):
    """Extract image parts from OpenAI-style content lists.

    Returns list of (bytes, mime_type). URL images are returned as
    (url_str, mime) placeholders and resolved by the provider.
    """
    images = []
    for msg in messages or []:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            t = c.get("type", "")
            if t == "image_url":
                u = c.get("image_url", {})
                url = u.get("url") if isinstance(u, dict) else u
                if isinstance(url, str) and url:
                    images.append((url, u.get("mime_type") if isinstance(u, dict) else None))
            elif t in ("image", "input_image"):
                u = c.get("image_url") or c.get("url")
                if isinstance(u, dict):
                    url = u.get("url")
                    if isinstance(url, str) and url:
                        images.append((url, u.get("mime_type")))
                elif isinstance(u, str) and u:
                    images.append((u, c.get("mime_type")))
                data = c.get("data") or c.get("base64")
                if isinstance(data, str):
                    mime = c.get("mime_type") or c.get("media_type") or "image/png"
                    images.append(("data:" + mime + ";base64," + data, mime))
    return images


def _decode_data_url(url):
    m = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", url, re.DOTALL)
    if not m:
        return None
    mime = m.group(1) or "image/png"
    is_b64 = bool(m.group(2))
    data = m.group(3)
    try:
        if is_b64:
            return base64.b64decode(data, validate=True), mime
        return unquote_to_bytes(data), mime
    except (ValueError, TypeError, binascii.Error):
        return None


def resolve_image(entry):
    """Resolve an image entry to (bytes, mime). Fetches URLs over HTTP."""
    if not isinstance(entry, tuple) or len(entry) != 2:
        return None
    src, mime = entry
    if isinstance(src, bytes):
        return src, mime or "image/png"
    if not isinstance(src, str):
        return None
    if src.startswith("data:"):
        return _decode_data_url(src)
    # remote URL
    mime = mime or "image/png"
    if src.startswith(("http://", "https://")):
        try:
            import urllib.request
            req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read(), mime
        except Exception:
            return None
    return None


def _tool_choice_instruction(tool_choice, tool_defs):
    if tool_choice == "none":
        return "\n\nIMPORTANT: Do NOT call any tools. Respond with text only."
    if tool_choice == "required":
        return "\n\nIMPORTANT: You MUST call at least one tool. Do not respond with text only."
    if isinstance(tool_choice, dict):
        fn_name = tool_choice.get("function", {}).get("name", "")
        if fn_name:
            return f'\n\nIMPORTANT: You MUST call the tool "{fn_name}". Do not call other tools.'
    return ""


def transcript(messages, tools=None, tool_choice=None):
    """Flatten OpenAI messages (+tools) into a single text prompt."""
    parts = []
    if tools and tool_choice != "none":
        tool_defs = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function", tool) if tool.get("type") == "function" else tool
            tool_defs.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        if tool_defs:
            constraint = _tool_choice_instruction(tool_choice, tool_defs)
            parts.append(
                "# Tool Use\n\n"
                "You can call the following tools. Call format:\n"
                '```tool_call\n{"name": "func_name", "arguments": {...}}\n```\n'
                "When calling tools, output ONLY the tool_call block(s).\n\n"
                f"Available tools:\n{json.dumps(tool_defs, ensure_ascii=False, indent=2)}"
                f"{constraint}"
            )

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") in ("text", "input_text", "output_text"):
                    text_parts.append(c.get("text", ""))
                else:
                    text_parts.append("[Image attached]")
            content = " ".join(text_parts)
        content = str(content)

        if role == "system":
            parts.append(f"[System instruction]: {content}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                tc_strs = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    tc_strs.append(
                        f'```tool_call\n{{"name": "{fn.get("name")}", '
                        f'"arguments": {fn.get("arguments", "{}")}}}\n```'
                    )
                parts.append(f"[Assistant]: {content or ''}\n" + "\n".join(tc_strs))
            else:
                parts.append(f"[Assistant]: {content}")
        elif role == "tool":
            parts.append(f"[Tool result for {msg.get('name', '')}]: {content}")
        else:
            parts.append(content if content else "")

    return "\n\n".join(p for p in parts if p)


def parse_tool_calls(text):
    """Extract ```tool_call blocks. Returns (clean_text, tool_calls_list)."""
    tool_calls = []
    pattern = r'```tool_call\s*\n(.*?)\n```'
    clean_parts = []
    last_end = 0
    for m in re.finditer(pattern, text, re.DOTALL):
        clean_parts.append(text[last_end:m.start()])
        last_end = m.end()
        try:
            data = json.loads(m.group(1).strip())
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False),
                },
            })
        except (json.JSONDecodeError, KeyError):
            pass
    clean_parts.append(text[last_end:])
    return "".join(clean_parts).strip(), tool_calls