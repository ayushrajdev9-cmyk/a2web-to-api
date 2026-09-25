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


def tool_instruction(tools, tool_choice="auto"):
    """Render the tool-use contract as standalone text.

    Used two ways: injected into a flattened transcript, or prepended as a
    system message for providers that keep native message arrays.
    """
    tool_defs = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool) if tool.get("type") == "function" else tool
        if not fn.get("name"):
            continue
        tool_defs.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    if not tool_defs:
        return ""
    constraint = _tool_choice_instruction(tool_choice, tool_defs)
    return (
        "# Tool Use\n\n"
        "You can call the following tools. Call format:\n"
        '```tool_call\n{"name": "func_name", "arguments": {...}}\n```\n'
        "When calling tools, output ONLY the tool_call block(s).\n\n"
        f"Available tools:\n{json.dumps(tool_defs, ensure_ascii=False, indent=2)}"
        f"{constraint}"
    )


def with_tool_instruction(messages, tools=None, tool_choice="auto"):
    """Prepend the tool contract as a system message (native-array providers).

    Never mutates the caller's messages.
    """
    msgs = list(messages or [])
    if not tools or tool_choice == "none":
        return msgs
    instr = tool_instruction(tools, tool_choice)
    if not instr:
        return msgs
    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "system":
            merged = dict(m)
            merged["content"] = f'{instr}\n\n{m.get("content", "")}'.strip()
            return msgs[:i] + [merged] + msgs[i + 1:]
    return [{"role": "system", "content": instr}] + msgs


def fold_tool_messages(messages):
    """Render assistant tool_calls and tool-role results as plain text.

    Providers that only accept system/user/assistant cannot carry tool messages,
    so a tool round-trip is folded into the conversation as readable text.
    """
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "tool":
            label = m.get("name") or m.get("tool_call_id") or ""
            out.append({"role": "user",
                        "content": f"[Tool result for {label}]: {content}"})
            continue
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = (tc or {}).get("function", {})
                calls.append(f'Called tool "{fn.get("name", "")}" with arguments '
                             f'{fn.get("arguments", "{}")}')
            if content:
                out.append({"role": "assistant", "content": str(content)})
            out.append({"role": "user", "content": "\n".join(calls)})
            continue
        out.append({"role": role, "content": content})
    return out


def transcript(messages, tools=None, tool_choice=None):
    """Flatten OpenAI messages (+tools) into a single text prompt."""
    parts = []
    if tools and tool_choice != "none":
        instr = tool_instruction(tools, tool_choice)
        if instr:
            parts.append(instr)

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


_TOOL_MARKER = "```tool_call"


def fence_pending(text: str) -> bool:
    """True while `text` may still turn into a ```tool_call block.

    Streaming holds text back so a raw tool block never leaks into content.
    Fences alternate open/close, so only the 0th, 2nd, 4th... open a block; if
    any of them opened (or may still open) a tool block, keep holding. A fence
    whose marker is clearly something else (```json, ```py) releases at once.
    """
    starts = [m.start() for m in re.finditer(r"```", text)]
    if not starts:
        # a trailing run of 1-2 backticks may still grow into a fence
        run = len(text) - len(text.rstrip("`"))
        return 1 <= run <= 2
    for start in starts[::2]:
        tail = text[start + 3:].split("\n", 1)[0]
        if tail == "" or _TOOL_MARKER[3:].startswith(tail):
            return True
    return False


def parse_tool_calls(text, strip=True):
    """Extract ```tool_call blocks. Returns (clean_text, tool_calls_list).

    strip=False keeps surrounding whitespace, which streaming needs when it
    re-sends the prose it held back — stripping there would drop characters
    the client never received.
    """
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
    clean = "".join(clean_parts)
    return (clean.strip() if strip else clean), tool_calls