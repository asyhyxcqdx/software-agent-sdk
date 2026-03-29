"""Check tool for task-maker depth validation."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from hashlib import sha1
from pathlib import Path

from pydantic import Field

from openhands.sdk import ImageContent, TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)


def _repo_slug(repo: str) -> str:
    return repo.replace("/", "_").lower()


def _slugify_test_file_path(test_file_path: str) -> str:
    normalized = test_file_path.strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
    safe = re.sub(r"_+", "_", safe).strip("_")
    readable = safe or "unknown_test_file"
    digest8 = sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{readable}-{digest8}"


def _parse_record_id(record_id: str) -> tuple[str, str, int]:
    match = re.match(r"^([^:]+/[^:]+)::(.+)::depth=(\d+)$", record_id.strip())
    if not match:
        raise ValueError(
            "Invalid record_id format. Expected: {repo}::{test_file_path}::depth={k}"
        )
    repo = match.group(1)
    test_file_path = match.group(2).strip()
    depth = int(match.group(3))
    return repo, test_file_path, depth


def _resolve_timeout_seconds() -> int:
    raw = os.getenv("RUN_TESTS_TIMEOUT")
    if raw is None or not raw.strip():
        raise ValueError("RUN_TESTS_TIMEOUT is missing")
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError("RUN_TESTS_TIMEOUT must be an integer") from exc
    if value <= 0:
        raise ValueError("RUN_TESTS_TIMEOUT must be > 0")
    return value


def _runtime_paths(repo: str, test_file_slug: str, depth: int) -> tuple[Path, Path]:
    repo_slug = _repo_slug(repo)
    run_tests_path = Path(f"/tmp/{repo_slug}/run_tests.py")
    check_dir = Path(f"/tmp/{repo_slug}/checks/{test_file_slug}/depth_{depth}")
    return run_tests_path, check_dir


def _next_check_index(check_dir: Path) -> int:
    max_index = 0
    for path in check_dir.glob("check_*.txt"):
        match = re.match(r"^check_(\d+)\.txt$", path.name)
        if not match:
            continue
        max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _parse_result_text_at_least_one_header(text: str) -> tuple[list[str], list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Invalid result format: empty result text")

    section: str | None = None
    saw_passed_header = False
    saw_failed_header = False
    passed_paths: list[str] = []
    failed_paths: list[str] = []

    for line in lines:
        lower_line = line.lower()
        if lower_line == "passed test files:":
            section = "passed"
            saw_passed_header = True
            continue
        if lower_line == "failed test files:":
            section = "failed"
            saw_failed_header = True
            continue

        if section == "passed":
            if not line.startswith("/testbed/"):
                raise ValueError(
                    "Invalid result format: passed paths must start with /testbed/"
                )
            passed_paths.append(line)
            continue

        if section == "failed":
            if not line.startswith("/testbed/"):
                raise ValueError(
                    "Invalid result format: failed paths must start with /testbed/"
                )
            failed_paths.append(line)
            continue

        # Ignore pre-header logs.
        continue

    if not (saw_passed_header or saw_failed_header):
        raise ValueError(
            "Invalid result format: expected at least one of passed/failed headers"
        )
    return sorted(set(passed_paths)), sorted(set(failed_paths))


class CheckAction(Action):
    record_id: str = Field(description="Canonical record id")
    repo: str = Field(description="Repository in owner/name format")
    depth: int = Field(ge=1, description="Current depth (baseline is depth=0)")
    f2p_candidates: list[str] = Field(
        default_factory=list,
        description="Candidate files for breakage target tracking",
    )
    p2p_candidates: list[str] = Field(
        default_factory=list,
        description="Candidate files for protection tracking",
    )


class CheckObservation(Observation):
    ok: bool
    message: str
    record_id: str
    depth: int
    f2p: list[str] = Field(default_factory=list)
    p2p: list[str] = Field(default_factory=list)

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        status = "OK" if self.ok else "ERROR"
        summary = [
            f"Check status: {status}",
            f"Message: {self.message}",
            f"record_id: {self.record_id}",
            f"depth: {self.depth}",
            f"f2p_count: {len(self.f2p)}",
            f"p2p_count: {len(self.p2p)}",
        ]
        return [TextContent(text="\n".join(summary))]


class CheckExecutor(ToolExecutor[CheckAction, CheckObservation]):
    def __call__(self, action: CheckAction, conversation=None) -> CheckObservation:  # noqa: ARG002
        if action.depth < 1:
            return CheckObservation(
                ok=False,
                message="depth must be >= 1",
                record_id=action.record_id,
                depth=action.depth,
                f2p=[],
                p2p=[],
            )

        try:
            parsed_repo, test_file_path, parsed_depth = _parse_record_id(action.record_id)
        except ValueError as exc:
            return CheckObservation(
                ok=False,
                message=f"invalid record_id: {exc}",
                record_id=action.record_id,
                depth=action.depth,
                f2p=[],
                p2p=[],
            )

        if parsed_repo != action.repo:
            return CheckObservation(
                ok=False,
                message="record_id repo does not match action.repo",
                record_id=action.record_id,
                depth=action.depth,
                f2p=[],
                p2p=[],
            )

        if parsed_depth != action.depth:
            return CheckObservation(
                ok=False,
                message="record_id depth does not match action.depth",
                record_id=action.record_id,
                depth=action.depth,
                f2p=[],
                p2p=[],
            )

        test_file_slug = _slugify_test_file_path(test_file_path)
        run_tests_path, check_dir = _runtime_paths(action.repo, test_file_slug, action.depth)
        if not run_tests_path.exists():
            return CheckObservation(
                ok=False,
                message=(
                    f"Missing run_tests.py at {run_tests_path}. "
                    "Ensure baseline(depth=0) has been prepared for this repo."
                ),
                record_id=action.record_id,
                depth=action.depth,
                f2p=[],
                p2p=[],
            )

        check_dir.mkdir(parents=True, exist_ok=True)
        round_index = _next_check_index(check_dir)
        result_path = check_dir / f"check_{round_index}.txt"

        cmd = [
            "python3",
            str(run_tests_path),
            "--input",
            "/testbed",
            "--output",
            str(result_path),
        ]
        try:
            timeout_seconds = _resolve_timeout_seconds()
        except ValueError as exc:
            return CheckObservation(
                ok=False,
                message=f"invalid timeout config: {exc}",
                record_id=action.record_id,
                depth=action.depth,
                f2p=[],
                p2p=[],
            )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CheckObservation(
                ok=False,
                message=f"check timeout after {timeout_seconds}s",
                record_id=action.record_id,
                depth=action.depth,
                f2p=[],
                p2p=[],
            )

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            detail_excerpt = detail[:1200]
            return CheckObservation(
                ok=False,
                message=(
                    f"run_tests.py failed with returncode={proc.returncode}. "
                    f"detail={detail_excerpt}"
                ),
                record_id=action.record_id,
                depth=action.depth,
                f2p=[],
                p2p=[],
            )

        if not result_path.exists():
            return CheckObservation(
                ok=False,
                message=f"result file missing: {result_path}",
                record_id=action.record_id,
                depth=action.depth,
                f2p=[],
                p2p=[],
            )

        try:
            passed_paths, failed_paths = _parse_result_text_at_least_one_header(
                result_path.read_text(encoding="utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            return CheckObservation(
                ok=False,
                message=f"invalid result format: {exc}",
                record_id=action.record_id,
                depth=action.depth,
                f2p=[],
                p2p=[],
            )

        f2p = sorted(set(failed_paths) & set(action.f2p_candidates))
        p2p = sorted(set(passed_paths) & set(action.p2p_candidates))
        return CheckObservation(
            ok=True,
            message=(
                f"check ok: depth={action.depth}, round={round_index}, "
                f"result={result_path}"
            ),
            record_id=action.record_id,
            depth=action.depth,
            f2p=f2p,
            p2p=p2p,
        )


class CheckTool(ToolDefinition[CheckAction, CheckObservation]):
    """Tool that checks current task-maker depth status."""

    @classmethod
    def create(cls, conv_state, **kwargs):  # noqa: ARG003
        return [
            cls(
                description=(
                    "Before calling this tool, you MUST ensure depth=0 baseline has "
                    "been prepared for this repo and your latest code edits are ready "
                    "for validation. "
                    "This tool executes run_tests.py and computes f2p/p2p against "
                    "candidates, which may be slow depending on test cost. "
                    "DO NOT call it repeatedly without meaningful code changes. "
                    "You MUST provide record_id, repo, depth, f2p_candidates, and "
                    "p2p_candidates. It reads timeout from RUN_TESTS_TIMEOUT and "
                    "writes per-round results under "
                    "/tmp/<repo_slug>/checks/<test_file_slug>/depth_<k>/check_<n>.txt."
                ),
                action_type=CheckAction,
                observation_type=CheckObservation,
                executor=CheckExecutor(),
            )
        ]


def register_check_tool() -> None:
    register_tool("check", CheckTool.create)


# Ensure tool is registered when module is imported on the client side
register_check_tool()
