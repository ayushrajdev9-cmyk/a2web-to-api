"""Shared utilities: logging, token estimation, SSE helpers."""

import json
import sys
import time
import uuid


def log(msg: str, enabled: bool = True, component: str = "a2web"):
    if not enabled:
        return
    sys.stderr.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{component}] {msg}\n")
    sys.stderr.flush()


def estimate_tokens(text: str) -> int:
    """Cheap ~4-chars-per-token estimator (no tokenizer dependency)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def usage_from(prompt: str, completion: str) -> dict:
    p = estimate_tokens(prompt)
    c = estimate_tokens(completion)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def now() -> int:
    return int(time.time())


def sse_chunk(chunk: dict) -> bytes:
    """Serialize one OpenAI-style SSE chunk."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()


def sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def chunk_first(cid: str, model: str) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": now(),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }


def chunk_delta(cid: str, model: str, delta: str) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": now(),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
    }


def chunk_end(cid: str, model: str, finish: str = "stop") -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": now(),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    }


def non_stream_response(cid: str, model: str, message: dict, finish: str,
                        prompt: str, completion: str) -> dict:
    return {
        "id": cid,
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage_from(prompt, completion),
    }


def iter_sse_lines(raw_stream):
    """Yield (event_name, data_str) from a byte SSE stream.

    Used by providers; data_str is the raw JSON payload of `data:` lines.
    """
    event = None
    data_parts = []
    for chunk in raw_stream:
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        for line in text.split("\n"):
            line = line.rstrip("\r")
            if line == "":
                if data_parts:
                    yield event, "\n".join(data_parts)
                event, data_parts = None, []
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_parts.append(line[5:].lstrip())
    if data_parts:
        yield event, "\n".join(data_parts)