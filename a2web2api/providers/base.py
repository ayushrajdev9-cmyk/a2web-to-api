"""Provider base class — the contract every web provider implements."""

import abc


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