"""Save tool precheck scaffold for task-maker depth snapshots."""

from __future__ import annotations

from collections.abc import Sequence
import json
import subprocess
from hashlib import sha1
from pathlib import Path

from pydantic import BaseModel, Field

from openhands.sdk import ImageContent, TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from alpha.shared.record_id_codec import (
    file_path_to_slug,
    parse_record_id,
    record_id_to_slug,
    repo_to_slug,
)


def _baseline_json_path(repo: str) -> Path:
    return Path(f"/tmp/{repo_to_slug(repo)}/baseline/baseline.json")


def _load_baseline_json(repo: str) -> dict:
    baseline_path = _baseline_json_path(repo)
    if not baseline_path.exists():
        raise ValueError(
            f"missing baseline file: {baseline_path}. run baseline(depth=0) first"
        )
    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid baseline json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid baseline json: root must be an object")
    baseline_pass_raw = payload.get("baseline_pass")
    if not isinstance(baseline_pass_raw, list):
        raise ValueError("invalid baseline json: baseline_pass must be a list")
    baseline_pass: list[str] = []
    for item in baseline_pass_raw:
        if not isinstance(item, str) or not item.startswith("/testbed/"):
            raise ValueError(
                "invalid baseline json: each baseline_pass item must start with /testbed/"
            )
        baseline_pass.append(item)
    commit0_raw = payload.get("commit0")
    if commit0_raw is None:
        payload["commit0"] = ""
    elif not isinstance(commit0_raw, str):
        raise ValueError("invalid baseline json: commit0 must be a string")
    payload["baseline_pass"] = sorted(set(baseline_pass))
    return payload


def _check_dir(repo: str, test_file_slug: str, depth: int) -> Path:
    repo_slug = repo_to_slug(repo)
    return Path(f"/tmp/{repo_slug}/checks/{test_file_slug}/depth_{depth}")


def _load_check_meta(meta_path: Path) -> dict:
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid check meta {meta_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid check meta {meta_path}: root must be an object")
    return payload


def _find_latest_success_meta(
    *, check_dir: Path, record_id: str, repo: str, depth: int
) -> dict | None:
    if not check_dir.exists():
        return None

    candidates: list[dict] = []
    for meta_path in check_dir.glob("check_*.meta.json"):
        payload = _load_check_meta(meta_path)
        if payload.get("ok") is not True:
            continue
        if payload.get("record_id") != record_id:
            continue
        if payload.get("repo") != repo:
            continue
        if payload.get("depth") != depth:
            continue
        round_raw = payload.get("round")
        if not isinstance(round_raw, int) or round_raw <= 0:
            raise ValueError(
                f"invalid check meta {meta_path}: round must be a positive integer"
            )
        workspace_fingerprint = payload.get("workspace_fingerprint")
        if not isinstance(workspace_fingerprint, str) or not workspace_fingerprint:
            raise ValueError(
                f"invalid check meta {meta_path}: workspace_fingerprint must be a non-empty string"
            )
        f2p_raw = payload.get("f2p")
        if not isinstance(f2p_raw, list) or any(
            not isinstance(item, str) for item in f2p_raw
        ):
            raise ValueError(f"invalid check meta {meta_path}: f2p must be list[str]")
        p2p_raw = payload.get("p2p")
        if not isinstance(p2p_raw, list) or any(
            not isinstance(item, str) for item in p2p_raw
        ):
            raise ValueError(f"invalid check meta {meta_path}: p2p must be list[str]")
        payload["workspace_fingerprint"] = workspace_fingerprint
        payload["f2p"] = sorted(set(f2p_raw))
        payload["p2p"] = sorted(set(p2p_raw))
        candidates.append(payload)

    if not candidates:
        return None
    return max(candidates, key=lambda item: item["round"])


def _git_run(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"git command failed: {' '.join(args)}: {exc}") from exc


def _compute_workspace_fingerprint() -> str:
    try:
        proc = subprocess.run(
            [
                "bash",
                "-lc",
                (
                    "set -euo pipefail; "
                    "git -C /testbed status --porcelain=v1 --untracked-files=all; "
                    "git -C /testbed diff --binary HEAD"
                ),
            ],
            capture_output=True,
            text=False,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"fingerprint command failed: {exc}") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"fingerprint command failed with returncode={proc.returncode}: {stderr}"
        )
    return sha1(proc.stdout or b"").hexdigest()


def _workspace_is_dirty() -> bool:
    proc = _git_run(
        [
            "-C",
            "/testbed",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git status failed: {detail}")
    return bool((proc.stdout or "").strip())


def _git_identity_effective() -> tuple[bool, str]:
    author = _git_run(["-C", "/testbed", "var", "GIT_AUTHOR_IDENT"])
    if author.returncode != 0:
        detail = (author.stderr or author.stdout or "").strip()
        return False, f"GIT_AUTHOR_IDENT unavailable: {detail}"
    committer = _git_run(["-C", "/testbed", "var", "GIT_COMMITTER_IDENT"])
    if committer.returncode != 0:
        detail = (committer.stderr or committer.stdout or "").strip()
        return False, f"GIT_COMMITTER_IDENT unavailable: {detail}"
    return True, ""


def _save_output_path(record_id: str, depth: int) -> Path:
    record_id_slug = record_id_to_slug(record_id)
    return Path(f"/output/{record_id_slug}/depth_{depth}/save.json")


def _write_save_output(task_record: "SaveTaskRecord") -> Path:
    output_path = _save_output_path(task_record.record_id, task_record.depth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = task_record.model_dump()
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


class SaveAction(Action):
    record_id: str = Field(description="Canonical record id")
    repo: str = Field(description="Repository in owner/name format")
    depth: int = Field(ge=1, description="Current depth (baseline is depth=0)")


class SaveTaskRecord(BaseModel):
    record_id: str
    repo: str
    commit: str
    f2p: list[str] = Field(default_factory=list)
    p2p: list[str] = Field(default_factory=list)
    gold_patch: str
    issue: str
    depth: int
    hint: str


class SaveObservation(Observation):
    ok: bool
    message: str
    task_record: SaveTaskRecord | None = None

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        status = "OK" if self.ok else "ERROR"
        summary = [
            f"Save status: {status}",
            f"Message: {self.message}",
        ]
        if self.task_record is not None:
            summary.append(f"record_id: {self.task_record.record_id}")
            summary.append(f"depth: {self.task_record.depth}")
            summary.append(f"commit: {self.task_record.commit}")
        return [TextContent(text="\n".join(summary))]


class SaveExecutor(ToolExecutor[SaveAction, SaveObservation]):
    @staticmethod
    def _error(message: str) -> SaveObservation:
        return SaveObservation(ok=False, message=message, task_record=None)

    @classmethod
    def _save_failed(cls, detail: str) -> SaveObservation:
        return cls._error(f"save_failed: {detail}")

    @classmethod
    def _post_commit_failure(cls, commit_k: str, detail: str) -> SaveObservation:
        return cls._error(
            "post_commit_failure: repo already mutated after commit; "
            f"commit_k={commit_k}; detail={detail}"
        )

    @staticmethod
    def _proc_detail(proc: subprocess.CompletedProcess[str]) -> str:
        return (proc.stderr or proc.stdout or "").strip()

    @classmethod
    def _reset_index_after_failure(cls) -> tuple[bool, str]:
        try:
            proc = _git_run(["-C", "/testbed", "reset"])
        except RuntimeError as exc:
            return False, str(exc)
        if proc.returncode != 0:
            return False, cls._proc_detail(proc)
        return True, ""

    @classmethod
    def _save_failed_with_reset(cls, detail: str) -> SaveObservation:
        reset_ok, reset_detail = cls._reset_index_after_failure()
        if reset_ok:
            return cls._save_failed(detail)
        return cls._save_failed(
            f"{detail}; rollback_failed: {reset_detail}; index may be left staged"
        )

    def _run_precheck(
        self, action: SaveAction
    ) -> tuple[str, dict] | SaveObservation:
        if action.depth < 1:
            return self._save_failed("depth must be >= 1")

        try:
            parsed_repo, test_file_path, parsed_depth = parse_record_id(action.record_id)
        except ValueError as exc:
            return self._save_failed(f"invalid record_id: {exc}")

        if parsed_repo != action.repo:
            return self._save_failed("record_id repo does not match action.repo")
        if parsed_depth != action.depth:
            return self._save_failed("record_id depth does not match action.depth")

        try:
            baseline = _load_baseline_json(action.repo)
        except ValueError as exc:
            return self._save_failed(f"invalid baseline state: {exc}")

        commit0 = str(baseline.get("commit0", "")).strip()
        baseline_pass = set(baseline["baseline_pass"])
        if not commit0:
            return self._save_failed("missing commit0 in baseline.json")
        if test_file_path not in baseline_pass:
            return self._save_failed("target test_file_path is not in baseline_pass")

        test_file_slug = file_path_to_slug(test_file_path)
        check_dir = _check_dir(action.repo, test_file_slug, action.depth)
        try:
            meta = _find_latest_success_meta(
                check_dir=check_dir,
                record_id=action.record_id,
                repo=action.repo,
                depth=action.depth,
            )
        except ValueError as exc:
            return self._save_failed(f"invalid check meta state: {exc}")

        if meta is None:
            return self._save_failed("no matching successful check metadata found")

        workspace_fingerprint = meta["workspace_fingerprint"]
        try:
            current_fingerprint = _compute_workspace_fingerprint()
        except RuntimeError as exc:
            return self._save_failed(f"failed to compute workspace fingerprint: {exc}")

        if current_fingerprint != workspace_fingerprint:
            return self._save_failed(
                "workspace changed after last successful check; rerun check_tool first"
            )

        try:
            is_dirty = _workspace_is_dirty()
        except RuntimeError as exc:
            return self._save_failed(str(exc))

        if not is_dirty:
            return self._save_failed("workspace clean; nothing to save")

        try:
            identity_ok, identity_msg = _git_identity_effective()
        except RuntimeError as exc:
            return self._save_failed(f"git identity check failed: {exc}")

        if not identity_ok:
            return self._save_failed(f"git identity missing: {identity_msg}")

        return commit0, meta

    def __call__(self, action: SaveAction, conversation=None) -> SaveObservation:  # noqa: ARG002
        precheck = self._run_precheck(action)
        if isinstance(precheck, SaveObservation):
            return precheck
        commit0, meta = precheck

        try:
            add_proc = _git_run(["-C", "/testbed", "add", "-A"])
        except RuntimeError as exc:
            return self._save_failed_with_reset(f"git add failed: {exc}")
        if add_proc.returncode != 0:
            return self._save_failed_with_reset(
                f"git add failed: {self._proc_detail(add_proc)}"
            )

        try:
            commit_proc = _git_run(
                [
                    "-C",
                    "/testbed",
                    "commit",
                    "-m",
                    f"task_maker save depth={action.depth}",
                ]
            )
        except RuntimeError as exc:
            return self._save_failed_with_reset(f"git commit failed: {exc}")
        if commit_proc.returncode != 0:
            return self._save_failed_with_reset(
                f"git commit failed: {self._proc_detail(commit_proc)}"
            )

        commit_k = "unknown"
        try:
            head_proc = _git_run(["-C", "/testbed", "rev-parse", "HEAD"])
        except RuntimeError as exc:
            return self._post_commit_failure(commit_k, f"rev-parse HEAD failed: {exc}")
        if head_proc.returncode != 0:
            return self._post_commit_failure(
                commit_k,
                f"rev-parse HEAD failed: {self._proc_detail(head_proc)}",
            )
        commit_k = (head_proc.stdout or "").strip()
        if not commit_k:
            return self._post_commit_failure(commit_k, "empty HEAD commit")

        try:
            diff_proc = _git_run(["-C", "/testbed", "diff", "--binary", commit0, commit_k])
        except RuntimeError as exc:
            return self._post_commit_failure(commit_k, f"git diff failed: {exc}")
        if diff_proc.returncode != 0:
            return self._post_commit_failure(
                commit_k,
                f"git diff failed: {self._proc_detail(diff_proc)}",
            )
        gold_patch = diff_proc.stdout or ""
        if not gold_patch.strip():
            return self._post_commit_failure(commit_k, "gold_patch is empty")

        task_record = SaveTaskRecord(
            record_id=action.record_id,
            repo=action.repo,
            commit=commit_k,
            f2p=meta["f2p"],
            p2p=meta["p2p"],
            gold_patch=gold_patch,
            issue="TODO(issue): placeholder",
            depth=action.depth,
            hint="TODO(hint): placeholder",
        )
        try:
            _write_save_output(task_record)
        except Exception as exc:  # noqa: BLE001
            return self._post_commit_failure(
                commit_k,
                f"failed to write save output: {exc}",
            )
        return SaveObservation(ok=True, message="save ok", task_record=task_record)


class SaveTool(ToolDefinition[SaveAction, SaveObservation]):
    """Tool that saves one depth snapshot into a task record."""

    @classmethod
    def create(cls, conv_state, **kwargs):  # noqa: ARG003
        return [
            cls(
                description=(
                    "Save current depth snapshot for task-maker. "
                    "Input requires record_id/repo/depth. "
                    "This version runs prechecks, commits workspace changes, and "
                    "produces gold_patch from diff(commit0, commit_k). It uses "
                    "post_commit_failure semantics if failures happen after commit. "
                    "Successful saves write /output/<record_id_slug>/depth_<k>/save.json. "
                    "Task DB persistence is performed by host-side task_maker ingest "
                    "(not inside save_tool). issue/hint are currently placeholder values."
                ),
                action_type=SaveAction,
                observation_type=SaveObservation,
                executor=SaveExecutor(),
            )
        ]


def register_save_tool() -> None:
    register_tool("save", SaveTool.create)


# Ensure tool is registered when module is imported on the client side
register_save_tool()
