"""Model routing: map a requested model name to (provider, upstream_id)."""


class ModelResolver:
    def __init__(self, providers: dict, default_model: str):
        self.providers = providers
        self.default_model = default_model
        # alias -> (provider_id, upstream_id, desc)
        self.alias_index = {}
        # upstream_id -> (provider_id, alias, desc)
        self.upstream_index = {}
        self._build_index()

    def _build_index(self):
        for pid, prov in self.providers.items():
            for alias, meta in (prov.models_lookup() or {}).items():
                upstream = meta.get("model", alias)
                desc = meta.get("desc", alias)
                if alias not in self.alias_index:
                    self.alias_index[alias] = (pid, upstream, desc)
                self.upstream_index.setdefault(upstream, (pid, alias, desc))
            if prov.default_model:
                upstream = prov.default_model
                desc = prov.models_lookup().get(prov.default_model, {}).get("desc", prov.default_model)
                if upstream not in self.alias_index:
                    self.alias_index[upstream] = (pid, upstream, desc)

    def all_models(self):
        """List of {"id", "object", "created", "owned_by", "description"}."""
        out = []
        for alias, (pid, upstream, desc) in sorted(self.alias_index.items()):
            out.append({
                "id": alias,
                "object": "model",
                "created": 1700000000,
                "owned_by": pid,
                "description": desc,
            })
        return out

    def resolve(self, name: str):
        """Return (provider_id, provider, upstream_id, meta, error).

        name may be:
          "alias"            — global alias or upstream id
          "provider/alias"   — explicit provider routing
        """
        if not name:
            return None, None, None, None, "model name required"

        if "/" in name:
            pid, rest = name.split("/", 1)
            prov = self.providers.get(pid)
            if not prov:
                return None, None, None, None, (
                    f"unknown provider '{pid}'. available: {', '.join(sorted(self.providers))}")
            meta = prov.models_lookup().get(rest) or {}
            upstream = meta.get("model", rest)
            return pid, prov, upstream, meta, None

        if name in self.alias_index:
            pid, upstream, desc = self.alias_index[name]
            return pid, self.providers[pid], upstream, {"desc": desc}, None

        # upstream id match (e.g. request "gpt-4o" instead of "chatgpt-4o")
        if name in self.upstream_index:
            pid, alias, desc = self.upstream_index[name]
            return pid, self.providers[pid], name, {"desc": desc}, None

        return None, None, None, None, (
            f"unknown model '{name}'. use provider/model e.g. 'deepseek/deepseek-chat', "
            f"or one of: {', '.join(sorted(self.alias_index))}")