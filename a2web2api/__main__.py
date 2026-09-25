"""CLI entry point: python -m a2web2api [options]"""

import argparse
import json
import sys

from . import __version__
from .config import load_config, find_config, DEFAULT_CONFIG


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m a2web2api",
        description="a2web2api — multi-provider web-to-API bridge (Ayush Rajdev & Anzar Iqbal)",
    )
    parser.add_argument("--host", default=None, help="bind host (default: config or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default: config or 8081)")
    parser.add_argument("--config", default=None, help="path to config.json")
    parser.add_argument("--proxy", default=None, help="proxy URL override (http://host:port)")
    parser.add_argument("--cookie-file", default=None,
                        help="default cookie file for providers without their own")
    parser.add_argument("--default-model", default=None, help="default model alias")
    parser.add_argument("--list-models", action="store_true",
                        help="print enabled models and exit")
    parser.add_argument("--check", action="store_true",
                        help="print provider status and exit")
    parser.add_argument("--version", action="version", version=f"a2web2api {__version__}")
    args = parser.parse_args(argv)

    config_path = args.config or find_config()
    cfg = load_config(config_path)
    if config_path:
        print(f"[a2web] config: {config_path}")

    if args.host:
        cfg["host"] = args.host
    if args.port:
        cfg["port"] = args.port
    if args.proxy:
        cfg["proxy"] = args.proxy
    if args.cookie_file:
        cfg["cookie_file"] = args.cookie_file
    if args.default_model:
        cfg["default_model"] = args.default_model

    from .providers import build_providers
    providers = build_providers(cfg)

    if args.list_models or args.check:
        if not providers:
            print("[a2web] no providers enabled")
            return 1
        for pid, prov in providers.items():
            models = ", ".join(prov.models_lookup().keys()) or "(none)"
            print(f"[{pid}] {prov.label}")
            print(f"    status : {prov.status()}")
            print(f"    default: {prov.default_model or '(none)'}")
            print(f"    models : {models}")
        return 0

    # write back effective config so server sees merged overrides
    from .server import serve
    serve(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())