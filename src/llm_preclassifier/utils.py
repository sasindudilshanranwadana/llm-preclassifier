import hashlib
import json
from typing import Any, Dict

def _flatten_text(content: Any) -> str:
    """Extract and normalize all text from an OpenAI-format message chunk."""
    if isinstance(content, str):
        return content.lower().strip()
    if isinstance(content, list):
        text = " ".join(
            item.get("text", "") 
            for item in content 
            if isinstance(item, dict) and item.get("type") == "text"
        )
        return text.lower().strip()
    return ""

def _has_tool_history(messages: list[Dict[str, Any]]) -> bool:
    """Check if the context includes any prior tool calls or results."""
    for msg in messages:
        if msg.get("role") == "tool" or msg.get("tool_calls"):
            return True
    return False

def _classification_cache_key(messages: list[Dict[str, Any]], max_turns: int = 5) -> str:
    """Generate a stable SHA-256 key for a message history slice."""
    relevant = messages[-max_turns:]
    canonical = json.dumps([
        {"r": m.get("role"), "c": _flatten_text(m.get("content", ""))} 
        for m in relevant
    ], separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
