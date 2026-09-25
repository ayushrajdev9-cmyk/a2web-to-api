#!/usr/bin/env bash
# Launch a2web-to-api (a2web2api) on port 8090
cd "$(dirname "$0")"
exec .venv/bin/python -m a2web2api --port 8090
