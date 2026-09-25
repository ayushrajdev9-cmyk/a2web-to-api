"""Provider factory — builds enabled providers from config."""

import copy

from .base import BaseProvider
from .gemini import GeminiProvider
from .deepseek import DeepSeekProvider
from .grok import GrokProvider
from .claude import ClaudeProvider
from .chatgpt import ChatGPTCookieProvider
from .custom import CustomProvider

BUILTIN_CLASSES = {
    "gemini": GeminiProvider,
    "deepseek": DeepSeekProvider,
    "grok": GrokProvider,
    "claude": ClaudeProvider,
    "chatgpt": ChatGPTCookieProvider,
}

#: providers that exist but need a cookie before they function at all
COOKIE_REQUIRED = {"deepseek", "grok", "claude", "chatgpt"}


def build_providers(cfg: dict) -> dict:
    """Return {provider_id: BaseProvider} for every enabled provider."""
    providers = {}

    for name, cls in BUILTIN_CLASSES.items():
        section = cfg.get("providers", {}).get(name)
        if isinstance(section, dict) and section.get("enabled") is False:
            continue
        merged = cfg["_provider"](name)
        # fall back to global cookie_file default
        if not merged.get("cookie_file") and cfg.get("cookie_file"):
            merged["cookie_file"] = cfg["cookie_file"]
        try:
            providers[name] = cls(merged)
        except Exception as e:
            import sys
            sys.stderr.write(f"[a2web] provider '{name}' failed to init: {e}\n")

    custom_cfg = cfg.get("providers", {}).get("custom", {})
    if isinstance(custom_cfg, dict) and custom_cfg.get("enabled") is False:
        return providers
    for pdef in (custom_cfg.get("providers") or []):
        if not isinstance(pdef, dict):
            continue
        pid = pdef.get("id")
        if not pid:
            continue
        merged = {
            "enabled": True,
            "proxy": cfg.get("proxy"),
            "log_requests": cfg.get("log_requests", True),
            "retry_attempts": cfg.get("retry_attempts", 3),
            "retry_delay_sec": cfg.get("retry_delay_sec", 2),
            "request_timeout_sec": cfg.get("request_timeout_sec", 180),
            "api_keys": cfg.get("api_keys", []),
        }
        merged.update(pdef)
        if not merged.get("cookie_file") and cfg.get("cookie_file"):
            merged["cookie_file"] = cfg["cookie_file"]
        try:
            providers[pid] = CustomProvider(merged)
        except Exception as e:
            import sys
            sys.stderr.write(f"[a2web] custom provider '{pid}' failed to init: {e}\n")

    return providers