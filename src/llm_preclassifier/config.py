"""Runtime configuration for llm-preclassifier."""
from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Settings:
    """Configuration with safe local-development defaults.

    Production mode requires at least one client API key. The service never
    persists request content; optional decision logs contain allowlisted
    decision metadata only.
    """

    environment: str = field(default_factory=lambda: os.getenv("LLM_PRECLASSIFIER_ENV", "development"))
    client_api_keys: str = field(default_factory=lambda: os.getenv("CLIENT_API_KEYS", ""))
    max_request_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_REQUEST_BYTES", "65536")))
    max_messages: int = field(default_factory=lambda: int(os.getenv("MAX_MESSAGES", "32")))
    cache_ttl_seconds: float = field(default_factory=lambda: float(os.getenv("CACHE_TTL_SECONDS", "300")))
    cache_max_entries: int = field(default_factory=lambda: int(os.getenv("CACHE_MAX_ENTRIES", "1024")))
    log_decisions: bool = field(default_factory=lambda: os.getenv("LOG_DECISIONS", "false").lower() == "true")
    decision_log_path: str = field(default_factory=lambda: os.getenv("DECISION_LOG_PATH", ""))
    semantic_model: str = field(default_factory=lambda: os.getenv("SEMANTIC_MODEL", ""))
    semantic_cache_dir: str = field(default_factory=lambda: os.getenv("SEMANTIC_CACHE_DIR", ""))
    policy_path: str = field(default_factory=lambda: os.getenv("POLICY_PATH", ""))

    @property
    def api_keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(key.strip() for key in self.client_api_keys.split(",") if key.strip()))

    def validate(self) -> None:
        if self.environment not in {"development", "production"}:
            raise RuntimeError("LLM_PRECLASSIFIER_ENV must be development or production")
        if self.max_request_bytes < 1024:
            raise RuntimeError("MAX_REQUEST_BYTES must be at least 1024")
        if self.max_messages < 1:
            raise RuntimeError("MAX_MESSAGES must be positive")
        if self.cache_max_entries < 1:
            raise RuntimeError("CACHE_MAX_ENTRIES must be positive")
        if self.environment == "production" and not self.api_keys:
            raise RuntimeError("CLIENT_API_KEYS must contain at least one key in production mode")
        if self.log_decisions and not self.decision_log_path:
            raise RuntimeError("DECISION_LOG_PATH is required when LOG_DECISIONS=true")
