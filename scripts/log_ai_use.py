"""PostToolUse hook: record every AI-assisted file edit into AI_USE.md.

The assignment requires declaring which files/functions AI helped with. Doing that
from memory at the end is unreliable; this appends a row automatically each time
an edit lands, so the declaration section writes itself.

Reads the hook payload as JSON on stdin. Never blocks: any failure here must not
interrupt the session, so all errors exit 0 silently.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

TABLE_HEADER = "| Date | File | Tool | Notes |"
TABLE_SEP = "| --- | --- | --- | --- |"


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    tool_name = payload.get("tool_name", "")
    if tool_name not in {"Edit", "Write", "NotebookEdit", "MultiEdit"}:
        return 0

    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not file_path:
        return 0

    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[1]))
    log_path = project_dir / "AI_USE.md"

    try:
        rel = str(Path(file_path).resolve().relative_to(project_dir.resolve()))
    except (ValueError, OSError):
        rel = str(file_path)

    # Don't log the log itself, or generated data.
    if rel in {"AI_USE.md"} or rel.startswith(("data", "results")):
        return 0

    row = f"| {datetime.now():%Y-%m-%d %H:%M} | `{rel}` | Claude Code ({tool_name}) | code generation |"

    try:
        text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        if TABLE_HEADER not in text:
            text += f"\n\n{TABLE_HEADER}\n{TABLE_SEP}\n"
        if row.rsplit("|", 2)[0] not in text:  # skip exact duplicates within a minute
            text = text.rstrip("\n") + "\n" + row + "\n"
        log_path.write_text(text, encoding="utf-8")
    except OSError:
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
