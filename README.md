# a2web-to-api

**`a2web2api` — multi-provider web-to-API bridge.**
Turns the web backends of **Gemini, DeepSeek, ChatGPT, Grok, Claude** and more into a single
**OpenAI-compatible** API server.

Made by **Ayush Rajdev & Anzar Iqbal**.

Built in the spirit of [gemini-web2api](https://github.com/Sophomoresty/gemini-web2api) — but
instead of one provider, it ships a provider layer so you can point one endpoint at whichever
model you want, and add new ones from config without writing code.

---

## Features

- **OpenAI-compatible** — `/v1/chat/completions`, `/v1/models`, `/v1/responses`
- **Streaming (SSE)** and non-streaming on every provider
- **Multi-provider** — Gemini, DeepSeek, ChatGPT, Grok, Claude built in
- **Extensible** — add *any* chat service via a config block (`providers.custom`), no code needed
- **Optional API keys** — OpenAI-style `Bearer` / `x-api-key` / `?key=` auth
- **Per-provider routing** — `deepseek/deepseek-chat`, `grok/grok-3`, `chatgpt/chatgpt-4o`, …
- **Multi-turn** — full message history is forwarded (per provider capability)
- **Tool calling + image input** where the provider supports it
- **Proxy aware** — global or per-provider, via config or env
- **Retries + backoff** per provider, streamed token deltas
- **Pure Python** — stdlib HTTP server, one dependency (`httpx`)

---

## Quick start

```bash
git clone <your-repo> a2web-to-api && cd a2web-to-api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp config.example.json config.json
# edit config.json: set cookie_file paths, api_keys, default_model

python -m a2web2api
```

Server starts on `http://0.0.0.0:8081/v1`.

Check what's live:

```bash
python -m a2web2api --check
curl http://localhost:8081/            # provider overview + status
curl http://localhost:8081/v1/models
```

---

## Client configuration

| Field    | Value                          |
| -------- | ------------------------------ |
| Base URL | `http://localhost:8081/v1`     |
| API Key  | any value in `api_keys` (or blank if empty) |
| Model    | `deepseek-chat`, `grok-4`, `chatgpt-4o`, `claude-sonnet-4`, `gemini-3.6-flash`, … |

### curl

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Hello!"}]}'
```

### Streaming

```bash
curl -N http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{"model":"grok-4","messages":[{"role":"user","content":"Write a haiku about APIs"}],"stream":true}'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8081/v1", api_key="sk-your-key")

resp = client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[{"role": "user", "content": "Explain websockets in 3 bullets"}],
)
print(resp.choices[0].message.content)
```

---

## Providers

| Provider  | Model prefix | Default model         | Cookie required | Notes |
| --------- | ------------ | --------------------- | --------------- | ----- |
| Gemini    | `gemini/`    | `gemini-3.6-flash`    | no (anonymous works) | StreamGenerate protocol; images + tools |
| DeepSeek  | `deepseek/`  | `deepseek-chat`       | yes | native OpenAI-format web API |
| ChatGPT   | `chatgpt/`   | `chatgpt-4o`          | yes | anti-bot may 403; see notes |
| Grok      | `grok/`      | `grok-4`              | yes | `grok.com/rest/app-chat` |
| Claude    | `claude/`    | `claude-sonnet-4`     | yes | `claude.ai/api/chat`, org auto-discovered |
| custom    | *(your id)*  | *(you define)*        | varies | config-driven; add anything |

Model names can be addressed two ways:

```
"deepseek-chat"          → global alias (unique across providers)
"deepseek/deepseek-chat" → explicit provider routing
```

### Gemini extras

- `gemini-3.5-flash-thinking` for extended thinking
- thinking depth override: append `@think=N` (0 = deepest … 4 = shallowest), e.g. `gemini-3.5-flash-thinking@think=0`
- `gemini-3.1-pro` for Pro routing (requires a cookie with a paid Gemini account for *real* Pro)
- image input via OpenAI `image_url` parts (http(s) URL or `data:` URL)
- tool calling (OpenAI `tools` format)

### Cookie export

For every provider except Gemini (which works anonymously) you need the cookies of a logged-in
session.

1. Open the site in Chrome and log in.
2. DevTools → Application → Cookies → the site domain.
3. Copy cookies as a single string: `K=V; K2=V2; ...`
4. Save to the path in `config.json` → `providers.<name>.cookie_file`.

Two formats are accepted:

```
# plain (K=V; K2=V2)
SID=xxx; HSID=yyy; SSID=zzz

# JSON (also carries extras)
{"cookie": "SID=xxx; HSID=yyy", "sapisid": "yyy", "xsrf_token": "AOOh0P..."}
```

Helper:

```bash
python scripts/cookie_helper.py convert netscape_export.txt cookies/grok.txt
python scripts/cookie_helper.py check cookies/grok.txt
```

---

## Configuration

Full reference in `config.example.json`.

```json
{
  "port": 8081,
  "api_keys": ["sk-your-key"],
  "proxy": null,
  "retry_attempts": 3,
  "default_model": "deepseek-chat",
  "providers": {
    "deepseek": { "enabled": true, "cookie_file": "cookies/deepseek.txt" },
    "grok":     { "enabled": true, "cookie_file": "cookies/grok.txt" }
  }
}
```

Global keys apply to every provider unless overridden in that provider's block.
When `api_keys` is `[]`, auth is disabled. CLI flags `--port/--host/--proxy/--cookie-file/--default-model`
override the file.

### Adding a provider from config only

```json
"custom": {
  "enabled": true,
  "providers": [
    {
      "id": "perplexity",
      "label": "Perplexity",
      "models": { "perplexity-sonar": { "model": "sonar", "desc": "Sonar" } },
      "endpoint": "https://www.perplexity.ai/rest/chat/answer",
      "method": "POST",
      "headers": { "Content-Type": "application/json" },
      "body": { "query": "{transcript}", "model": "{model}", "attachments": [] },
      "response": { "mode": "sse", "data_path": "answer", "final_text_path": "answer" }
    }
  ]
}
```

Body placeholders: `{prompt}` (last user text), `{transcript}` (flattened conversation),
`{model}` (upstream id), and `{"$json": "messages"}` for a raw message array.
Response `data_path` / `final_text_path` are dotted paths (`choices.0.delta.content`) into the
JSON payload; `mode: "sse"` streams, `mode: "json"` is one-shot.

---

## Endpoints

| Method | Path                     | Purpose |
| ------ | ------------------------ | ------- |
| GET    | `/`                      | status + provider overview |
| GET    | `/health`                | liveness |
| GET    | `/v1/models`             | model list |
| POST   | `/v1/chat/completions`   | chat (stream + non-stream) |
| POST   | `/v1/responses`          | Responses API (Codex-style) |

---

## Docker

```bash
cp config.example.json config.json
docker build -t a2web2api .
docker run -d --name a2web2api -p 8081:8081 \
  -v ./config.json:/app/config.json:ro \
  -v ./cookies:/app/cookies:ro a2web2api
```

or `docker compose up -d`.

If upstreams return empty/rejected responses through Docker's NAT, use host networking
(`network_mode: host`) or set `proxy`.

---

## Testing

```bash
pip install pytest
python -m pytest tests/ -q       # offline: config, providers, routing, server pipeline
```

---

## Notes & limitations

- **Stateless single-turn emphasis** — history is forwarded per provider capability; some web
  endpoints flatten to one prompt.
- **Web UIs change.** Endpoints and model names drift upstream; model tables in each provider are
  config-overridable so you can patch without code.
- **ChatGPT anti-bot** (Arkose/Turnstile, Cloudflare) can reject cookie-only access from
  datacenter IPs — a residential proxy helps.
- **Tool calling / image input** are implemented where the provider supports it; the server returns
  a clear `400` otherwise.
- Use within the terms of the respective services. This project does not circumvent authentication —
  you supply your own session cookies for accounts you own.

---

## License

MIT — see `LICENSE`.

© 2026 Ayush Rajdev & Anzar Iqbal
