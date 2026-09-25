"""Provider base class — the contract every web provider implements."""

import abc


class AuthRequired(ValueError):
    """A cookie-gated provider was called without a usable session.

    Subclasses ValueError so the server surfaces it as a client-side 400 with
    an actionable message rather than an opaque upstream parse failure.
    """


class BaseProvider(abc.ABC):
    """A chat provider backed by a web service.

    Subclasses implement `chat()` which returns a generator of text deltas
    (streaming) or a plain string (non-streaming). The server wrapper turns
    whatever is returned into OpenAI-format output.
    """

    #: canonical provider id, e.g. "deepseek"
    name = "base"
    #: human label
    label = "Base"
    #: {alias: {"model": upstream_id, "desc": "..."}} — aliases exposed via /v1/models
    models = {}
    default_model = ""
    #: path to a session-cookie file; None means the provider works anonymously
    cookie_file = None

    def __init__(self, cfg: dict):
        # cfg: merged global + provider section; helpers attached by factory
        self.cfg = cfg
        self.log_enabled = cfg.get("log_requests", True)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def status(self) -> str:
        """Return an informational status string shown at startup and /health."""
        return "config loaded"

    def close(self):
        """Release any resources (sessions, clients)."""

    # ── helpers ────────────────────────────────────────────────────────────

    def log(self, msg: str):
        from ..util import log
        log(msg, enabled=self.log_enabled, component=self.name)

    def models_lookup(self) -> dict:
        """Map of exposed alias -> {"model": upstream, "desc": ...}."""
        return dict(self.models)

    def require_cookie(self):
        """Fail fast when a cookie-gated provider has no session.

        Without this the request goes out unauthenticated and comes back as an
        HTML login page, which surfaces as a baffling JSON parse error.
        """
        from ..cookie import load_cookie
        cookie_str, _ = load_cookie(self.cookie_file)
        if not cookie_str:
            raise AuthRequired(
                f"provider '{self.name}' requires a session cookie: log in, export the "
                f"cookies, and set providers.{self.name}.cookie_file in config.json "
                f"(currently: {self.cookie_file or 'not set'})"
            )
        return cookie_str

    # ── core ───────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def chat(self, messages, model: str, stream: bool = False, **kw):
        """Generate a reply.

        Args:
            messages: normalized OpenAI-style message list
            model:    upstream model id (already resolved from alias)
            stream:   if True return a generator of text deltas;
                      otherwise return the full text string.
        """
        raise NotImplementedError

    # ── optional capabilities ──────────────────────────────────────────────

    def supports_tools(self) -> bool:
        return False

    def supports_images(self) -> bool:
        return False