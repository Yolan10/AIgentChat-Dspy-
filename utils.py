"""Utility functions for timestamping and file I/O."""
import json
import os
from datetime import datetime, timezone

import config


def get_timestamp() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def ensure_logs_dir():
    os.makedirs(config.LOGS_DIRECTORY, exist_ok=True)


def save_conversation_log(log_obj: dict, filename: str) -> None:
    """Save a conversation log as JSON under the logs directory."""
    ensure_logs_dir()
    path = os.path.join(config.LOGS_DIRECTORY, filename)
    with open(path, "w", encoding="utf-8") as f:
        # Use default=str so any non-serializable objects are converted to
        # strings rather than raising an exception.
        json.dump(log_obj, f, indent=config.JSON_INDENT, default=str)


def load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_template(template_str: str, variables: dict) -> str:
    text = template_str
    for key, val in variables.items():
        text = text.replace(f"{{{{{key}}}}}", str(val))
    return text


def extract_json_array(text: str):
    """Return the first JSON array found in the provided text."""
    import re

    match = re.search(r"\[[^\[]*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None
