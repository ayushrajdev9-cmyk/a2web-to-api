"""Configuration management: global defaults + per-provider sections."""

import json
import os
import copy

DEFAULT_CONFIG = {
    "host": "0.0.0.0",
    "port": 8081,
    "api_keys": [],                 # empty list => auth disabled; else Bearer/x-api-key/query key
    "proxy": None,                  # global proxy, e.g. http://127.0.0.1:7890
    "log_requests": True,
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 180,
    "cookie_file": None,            # default cookie file for providers that don't set their own
    "default_model": "deepseek-chat",
    "providers": {},                # provider name -> section; enabled providers are registered
}


def _merge_provider_defaults(cfg: dict, provider_name: str) -> dict:
    """Merge global defaults into a provider section (provider wins)."""
    section = cfg.get("providers", {}).get(provider_name, {})
    merged = {
        "enabled": True,
        "proxy": cfg.get("proxy"),
        "log_requests": cfg.get("log_requests", True),
        "retry_attempts": cfg.get("retry_attempts", 3),
        "retry_delay_sec": cfg.get("retry_delay_sec", 2),
        "request_timeout_sec": cfg.get("request_timeout_sec", 180),
        "api_keys": cfg.get("api_keys", []),
    }
    if isinstance(section, dict):
        merged.update(section)
    return merged


def load_config(path: str = None) -> dict:
    """Load config from JSON file (if present) over top of defaults.

    Returns the full merged config dict.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        if not isinstance(user_cfg, dict):
            raise ValueError("config file must contain a JSON object")
        for k, v in user_cfg.items():
            if k == "providers" and isinstance(v, dict):
                cfg["providers"].update(v)
            else:
                cfg[k] = v

    # per-provider merge helper attached for provider factory use
    cfg["_provider"] = lambda name: _merge_provider_defaults(cfg, name)
    return cfg


def find_config() -> str:
    """Search standard config locations."""
    for p in ["./config.json",
              os.path.expanduser("~/.config/a2web2api/config.json")]:
        if os.path.exists(p):
            return p
    return None