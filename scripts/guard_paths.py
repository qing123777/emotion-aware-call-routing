"""PreToolUse hook: refuse writes that would break the submission rules.

Two rules, both from the assignment brief:
  1. Nothing may be written into data/ (datasets live on Drive, not in the repo).
  2. No file over 5MB may be created in the repo (the whole zip is capped at 5MB).

Exit code 2 blocks the tool call and returns the message on stderr to Claude.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
BLOCKED_PREFIXES = ("data/", "data\\")
BLOCKED_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".wav", ".mp4")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path")
    if not file_path:
        return 0

    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[1]))
    try:
        rel = Path(file_path).resolve().relative_to(project_dir.resolve())
    except (ValueError, OSError):
        return 0  # outside the project: not our business

    rel_str = str(rel).replace("\\", "/")

    if rel_str.startswith("data/") and not rel_str.startswith("data/processed/"):
        print(
            f"Blocked: {rel_str} is under data/. Datasets and raw audio must not enter the repo "
            "(5MB submission cap). Derived manifests belong in data/processed/.",
            file=sys.stderr,
        )
        return 2

    if rel.suffix.lower() in BLOCKED_SUFFIXES:
        print(
            f"Blocked: {rel_str} has extension {rel.suffix} (model weights or media). "
            "Save these to Google Drive and link them in README.md instead.",
            file=sys.stderr,
        )
        return 2

    content = tool_input.get("content") or tool_input.get("new_string") or ""
    if len(content.encode("utf-8", errors="ignore")) > MAX_BYTES:
        print(f"Blocked: writing {rel_str} would exceed the 5MB submission cap.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
