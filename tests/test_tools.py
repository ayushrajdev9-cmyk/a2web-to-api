"""Tests for prompt-emulated tool calling (providers without native tool support).

No network calls. Run:  python -m pytest tests/  or  python tests/test_tools.py

Made by Ayush Rajdev & Anzar Iqbal.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2web2api.prompt import (
    fence_pending, fold_tool_messages, parse_tool_calls, tool_instruction,
    transcript, with_tool_instruction,
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]


def test_tool_instruction():
    instr = tool_instruction(TOOLS)
    assert "# Tool Use" in instr
    assert "get_weather" in instr
    assert "```tool_call" in instr
    assert tool_instruction([]) == ""
    assert tool_instruction(None) == ""
    # a tool without a name is skipped
    assert tool_instruction([{"type": "function", "function": {}}]) == ""


def test_tool_choice_constraints():
    assert "MUST call at least one tool" in tool_instruction(TOOLS, "required")
    assert 'MUST call the tool "get_weather"' in tool_instruction(
        TOOLS, {"function": {"name": "get_weather"}})
    print("PASS tool_choice_constraints")


def test_with_tool_instruction_merges_system():
    msgs = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}]
    out = with_tool_instruction(msgs, TOOLS)
    assert len(out) == 2, out
    assert out[0]["role"] == "system"
    assert "# Tool Use" in out[0]["content"] and "Be terse." in out[0]["content"]
    # no system message -> prepend one
    out2 = with_tool_instruction([{"role": "user", "content": "hi"}], TOOLS)
    assert out2[0]["role"] == "system" and "# Tool Use" in out2[0]["content"]
    # tool_choice=none -> untouched
    assert len(with_tool_instruction(msgs, TOOLS, "none")) == 2
    assert "# Tool Use" not in with_tool_instruction(msgs, TOOLS, "none")[0]["content"]
    print("PASS with_tool_instruction")


def test_fold_tool_messages():
    loop = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}}]},
        {"role": "tool", "name": "get_weather", "tool_call_id": "c1", "content": "18C sunny"},
    ]
    folded = fold_tool_messages(loop)
    assert all(m["role"] in ("system", "user", "assistant") for m in folded), folded
    blob = json.dumps(folded)
    assert "get_weather" in blob and "18C sunny" in blob, folded
    # plain messages pass through unchanged
    plain = [{"role": "user", "content": "hi"}]
    assert fold_tool_messages(plain) == plain
    print("PASS fold_tool_messages")


def test_transcript_includes_tools():
    loop = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}}]},
        {"role": "tool", "name": "get_weather", "content": "18C sunny"},
    ]
    text = transcript(loop, TOOLS, "auto")
    assert "# Tool Use" in text
    assert "18C sunny" in text, text
    assert "# Tool Use" not in transcript(loop, TOOLS, "none")
    print("PASS transcript_includes_tools")


def test_parse_tool_calls():
    raw = ('Sure thing.\n```tool_call\n'
           '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n```')
    clean, calls = parse_tool_calls(raw)
    assert clean == "Sure thing.", clean
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[0]["function"]["arguments"] == '{"city": "Tokyo"}'
    assert calls[0]["id"].startswith("call_")
    assert calls[0]["type"] == "function"
    # no block -> unchanged text, no calls
    assert parse_tool_calls("just text") == ("just text", [])
    # malformed block is dropped, not fatal
    assert parse_tool_calls("```tool_call\nnot json\n```")[1] == []
    # several blocks -> several calls
    two = ('```tool_call\n{"name":"a","arguments":{}}\n```\n'
           '```tool_call\n{"name":"b","arguments":{}}\n```')
    assert len(parse_tool_calls(two)[1]) == 2
    print("PASS parse_tool_calls")


def test_fence_pending():
    # a tool fence (or a partial one) must be held back
    for s in ("`", "``", "```", "```tool", "```tool_call", "```tool_call\n{",
              'done\n```tool', "```tool_cal"):
        assert fence_pending(s) is True, s
    # anything clearly not a tool fence releases immediately
    for s in ("plain text", "```json\n{}", "```py\ncode", "a ```b", "```toolx"):
        assert fence_pending(s) is False, s
    # a *closing* fence is recognised because earlier fences are still in text
    assert fence_pending("```py\ncode\n```") is False
    assert fence_pending("```py\ncode\n```\ndone") is False
    print("PASS fence_pending")


def _replay(deltas):
    """Mirror the server's streaming hold logic; returns (content, tool_calls)."""
    seen, emitted, content = "", 0, ""
    for d in deltas:
        if not d:
            continue
        seen += d
        if not fence_pending(seen):
            content += seen[emitted:]
            emitted = len(seen)
    clean, calls = parse_tool_calls(seen)
    if calls:
        # prose trapped in the held tail is still owed to the client
        owed, _ = parse_tool_calls(seen[emitted:], strip=False)
        extra = owed or None
    else:
        # server flushes whatever was still held, then finishes with "stop"
        content += seen[emitted:]
        extra = None
    return content, calls, extra


def test_stream_tool_call_never_leaks():
    raw = ('```tool_call\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n```')
    for size in (1, 2, 3, 5, 7, 12, 40, len(raw)):
        deltas = [raw[i:i + size] for i in range(0, len(raw), size)]
        content, calls, _ = _replay(deltas)
        assert content == "", (size, content)
        assert len(calls) == 1, (size, calls)
        assert calls[0]["function"]["name"] == "get_weather"
    print("PASS stream_tool_call_never_leaks")


def test_stream_normal_text_intact():
    for deltas in (["Hello", ", ", "world", "!"],
                   ["Here:\n", "```py\n", "print(1)\n", "```\n", "ok"],
                   ["Data:\n", "```json\n", '{"a":1}\n', "```"],
                   ["use `x` and `y` here"],
                   ["text\n", "```\n", "still going"]):
        content, calls, _ = _replay(deltas)
        assert content == "".join(deltas), deltas
        assert calls == [], deltas
    print("PASS stream_normal_text_intact")


def test_stream_prose_then_tool_call():
    deltas = ["Let me ", "check.\n```too", 'l_call\n{"name": "f"',
              ', "arguments": {}}\n```']
    content, calls, extra = _replay(deltas)
    assert calls and calls[0]["function"]["name"] == "f", calls
    assert "```" not in content, content
    # streamed prose + owed prose must reconstruct the prose exactly once
    assert (content + (extra or "")).strip() == "Let me check.", (content, extra)


def test_stream_prose_split_at_fence_boundary():
    """Prose already streamed must not be re-sent in the tool_calls chunk.

    The fence can open mid-prose, leaving a fragment (e.g. ".") in the held
    tail; the client gets each character exactly once.
    """
    deltas = ["Let me lo", "ok that u",
              'p.\n```tool_call\n{"name": "f", "arguments": {}}\n```']
    content, calls, extra = _replay(deltas)
    assert calls and len(calls) == 1, calls
    assert "```" not in content, content
    assert (content + (extra or "")).strip() == "Let me look that up.", (content, extra)
    # no character may be delivered twice
    assert content.count(".") + (extra or "").count(".") == 1


def test_stream_prose_never_duplicated_any_chunk_size():
    """Whatever the chunking, streamed + owed prose equals the prose once."""
    raw = ('Let me look that up.\n```tool_call\n'
           '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n```')
    for size in (1, 2, 3, 4, 7, 11, 23, len(raw)):
        deltas = [raw[i:i + size] for i in range(0, len(raw), size)]
        content, calls, extra = _replay(deltas)
        assert len(calls) == 1, (size, calls)
        assert "```" not in content, (size, content)
        # whitespace around the fence may legitimately stream before it is seen,
        # but no prose may be delivered twice
        assert (content + (extra or "")).strip() == "Let me look that up.", \
            (size, content, extra)
    print("PASS stream_prose_then_tool_call")


def test_server_accepts_tools_for_non_native_provider():
    """A provider without native tool support must NOT be rejected."""
    import json
    import threading
    import urllib.error
    import urllib.request

    from a2web2api.server import A2WebHandler, ThreadingHTTPServerV2

    class StubProvider:
        name = "stub"
        label = "Stub"
        default_model = "stub-1"

        def __init__(self, reply):
            self.cfg = {}
            self.reply = reply
            self.seen = {}

        def models_lookup(self):
            return {"stub-1": {"model": "stub-1", "desc": "stub"}}

        def status(self):
            return "stub"

        def supports_tools(self):
            return False          # no native tool support

        def supports_images(self):
            return False

        def chat(self, messages, model, stream=False, tools=None, tool_choice=None, **kw):
            self.seen = {"messages": messages, "tools": tools, "tool_choice": tool_choice}
            text = self.reply
            if stream:
                return (text[i:i + 7] for i in range(0, len(text), 7))
            return text

        def close(self):
            pass

    class ResolverStub:
        def __init__(self, prov):
            self.providers = {"stub": prov}

        def resolve(self, name):
            return "stub", self.providers["stub"], "stub-1", {"desc": "s"}, None

        def all_models(self):
            return [{"id": "stub-1", "object": "model", "created": 0,
                     "owned_by": "stub", "description": "stub"}]

    tool_reply = ('```tool_call\n{"name": "get_weather", '
                  '"arguments": {"city": "Tokyo"}}\n```')
    prov = StubProvider(tool_reply)
    server = ThreadingHTTPServerV2(("127.0.0.1", 0), A2WebHandler)
    server.providers = {"stub": prov}
    server.resolver = ResolverStub(prov)
    server.api_keys = []
    server.default_model = "stub-1"
    server.log_requests = False
    server.cfg = {}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def post(payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=10).read().decode()

    payload = {"model": "stub-1", "messages": [{"role": "user", "content": "weather?"}],
               "tools": TOOLS}

    # non-stream: 200 with synthesized tool_calls (was a 400 before)
    resp = json.loads(post(payload))
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls", resp
    tc = choice["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather", resp
    assert tc["function"]["arguments"] == '{"city": "Tokyo"}', resp
    # the provider still received the tools so it can emulate them
    assert prov.seen["tools"] == TOOLS, prov.seen

    # stream: raw block must not appear in content, tool_calls chunk instead
    raw = post(dict(payload, stream=True))
    assert "data: [DONE]" in raw
    content, calls = "", []
    for line in raw.split("\n"):
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            continue
        obj = json.loads(line[6:])
        for ch in obj.get("choices", []):
            d = ch.get("delta") or {}
            if d.get("content"):
                content += d["content"]
            if d.get("tool_calls"):
                calls.extend(d["tool_calls"])
            if ch.get("finish_reason") == "tool_calls":
                assert "```" not in content, content
    assert content == "", content
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather", raw

    # tool_choice=none must not force emulation and must stream text as-is
    prov.reply = "plain answer"
    raw = post(dict(payload, stream=True, tool_choice="none"))
    streamed = ""
    for line in raw.split("\n"):
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            continue
        for ch in json.loads(line[6:]).get("choices", []):
            streamed += (ch.get("delta") or {}).get("content") or ""
    assert streamed == "plain answer", raw
    assert "tool_calls" not in raw, raw

    server.shutdown()
    print("PASS server_accepts_tools_for_non_native_provider")


import json  # noqa: E402  (used by the helper above)

if __name__ == "__main__":
    test_tool_instruction()
    test_tool_choice_constraints()
    test_with_tool_instruction_merges_system()
    test_fold_tool_messages()
    test_transcript_includes_tools()
    test_parse_tool_calls()
    test_fence_pending()
    test_stream_tool_call_never_leaks()
    test_stream_normal_text_intact()
    test_stream_prose_then_tool_call()
    test_stream_prose_split_at_fence_boundary()
    test_stream_prose_never_duplicated_any_chunk_size()
    test_server_accepts_tools_for_non_native_provider()
    print("ALL TOOL TESTS PASSED")
