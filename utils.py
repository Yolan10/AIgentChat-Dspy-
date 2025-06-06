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


def _run_counter_path() -> str:
    """Return the path of the run counter file."""
    ensure_logs_dir()
    return os.path.join(config.LOGS_DIRECTORY, "run_counter.txt")


def get_run_number() -> int:
    """Return the current run number stored on disk (0 if not found)."""
    path = _run_counter_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except Exception:
        return 0


def increment_run_number() -> int:
    """Increment and persist the run counter, returning the new value."""
    run_no = get_run_number() + 1
    path = _run_counter_path()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(str(run_no))
    return run_no


def format_agent_id(run_no: int, index: int) -> str:
    """Return a unique agent identifier for the given run and index."""
    ts = get_timestamp().replace(":", "").replace("-", "")
    return f"{run_no}.{index}_{ts}"



def _wrap_text(text: str, width: int = 150) -> str:
    """Return text wrapped with newline every `width` characters."""
    lines = []
    for line in text.splitlines():
        while len(line) > width:
            lines.append(line[:width])
            line = line[width:]
        lines.append(line)
    return "\n".join(lines)


def append_improvement_log(run_no: int, prompt: str) -> None:
    """Append the improved prompt to a persistent text log."""
    ensure_logs_dir()
    path = os.path.join(config.LOGS_DIRECTORY, "improved_prompts.txt")
    ts = get_timestamp()
    wrapped = _wrap_text(prompt, 150)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{ts} run={run_no} instructions=\"{wrapped}\"\n")



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
