"""Validator tool for checking agent outputs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic import Field

from openhands.sdk import ImageContent, TextContent
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)


@dataclass
class ValidationResult:
    ok: bool
    message: str


class ValidatorAction(Action):
    dockerfile_path: str = Field(description="Path to Dockerfile")
    test_script_path: str = Field(description="Path to Python test script")
    extra_info_path: str = Field(description="Path to extra info JSON")


class ValidatorObservation(Observation):
    ok: bool
    message: str

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        status = "OK" if self.ok else "ERROR"
        summary = [
            f"Validator status: {status}",
            f"Message: {self.message}",
        ]
        return [TextContent(text="\n".join(summary))]


class ValidatorExecutor(ToolExecutor[ValidatorAction, ValidatorObservation]):
    def __call__(
        self,
        action: ValidatorAction,
        conversation=None,
    ) -> ValidatorObservation:  # noqa: ARG002
        legal_check = None
        parse_failure_message = None
        if conversation is not None and hasattr(conversation, "ask_agent"):
            legal_check, parse_failure_message = _detect_test_script_legal(
                conversation,
                action.test_script_path,
            )
        if legal_check is False:
            _set_extra_info_status(action.extra_info_path, "failed")
            return ValidatorObservation(
                ok=False,
                message="Illegal test script detected. Ensure run_tests.py passed/failed test files are generated from actual test execution and supports --input/--output.",
            )
        if parse_failure_message is not None:
            _set_extra_info_status(action.extra_info_path, "failed")
            return ValidatorObservation(
                ok=False,
                message=parse_failure_message,
            )
        
        result = request_host_validation(
            dockerfile_path=action.dockerfile_path,
            test_script_path=action.test_script_path,
            extra_info_path=action.extra_info_path,
        )
        _set_extra_info_status(
            action.extra_info_path, "success" if result.ok else "failed"
        )
        if result.ok and conversation is not None:
            conversation.state.execution_status = ConversationExecutionStatus.FINISHED
        return ValidatorObservation(
            ok=result.ok,
            message=result.message,
        )


class ValidatorTool(ToolDefinition[ValidatorAction, ValidatorObservation]):
    """Tool that validates builder artifacts by building and running tests."""

    @classmethod
    def create(cls, conv_state, **kwargs):  # noqa: ARG003
        return [
            cls(
                description=(
                    "Before calling this tool, you MUST successfully run your own "
                    "test runner locally and verify the result file format is "
                    "correct. This tool triggers a host-side image build and test "
                    "execution, which is expensive and slow. DO NOT call it unless "
                    "you are absolutely certain the local validation is correct."
                ),
                action_type=ValidatorAction,
                observation_type=ValidatorObservation,
                executor=ValidatorExecutor(),
            )
        ]


def _set_extra_info_status(extra_info_path: str, status: str) -> None:
    path = Path(extra_info_path)
    payload: dict[str, Any]
    try:
        payload = json.loads(path.read_text()) if path.exists() else {}
    except Exception:  # noqa: BLE001
        payload = {}
    payload["status"] = status
    path.write_text(json.dumps(payload, indent=2))


def _detect_test_script_legal(
    conversation, test_script_path: str
) -> tuple[bool | None, str | None]:
    try:
        test_script_text = Path(test_script_path).read_text()
    except Exception:  # noqa: BLE001
        return None, None

    base_prompt = "You are reviewing a test script used to produce passed/failed test files. You must judge two things:\n" \
        "1. Cheating: The script produces those lists without actually running tests (e.g., hardcoded lists, cached outputs, or placeholders). It also includes any case where files exist but were not produced by a real test run; if the script does not clearly run tests and derive the lists from results, treat it as cheating.\n" \
        "2. Input/output args: The run_tests.py script supports --input and --output arguments and uses them to read inputs and write outputs accordingly.\n\n" \
        f"Path: {test_script_path}\n" \
        f"Script:\n{test_script_text}\n\n" \
        "Return JSON only: {\"legal\": true/false, \"reason\": \"...\"}."
    retry_prompt = "You must judge two things:\n" \
        "1. Cheating: If the script does not clearly run tests and derive passed/failed lists from results, it is cheating. Files that exist but were not produced by a real test run also count as cheating.\n" \
        "2. Input/output args: The run_tests.py script supports --input and --output arguments and uses them to read inputs and write outputs accordingly.\n\n" \
        f"Path: {test_script_path}\n" \
        f"Script:\n{test_script_text}\n\n" \
        "Return JSON only. No extra text. Format: {\"legal\": true/false, \"reason\": \"...\"}."
    
    had_error = False
    for prompt in (base_prompt, retry_prompt, retry_prompt):
        try:
            response = conversation.ask_agent(prompt)
        except Exception:  # noqa: BLE001
            had_error = True
            continue
        parsed = _parse_legal_response(response)
        if parsed is not None:
            return parsed, None
    # Calling ask_agent() failed, returning a generic response
    if had_error:
        return (
            None,
            "please ensure the passed test files and failed test files in your test script are generated from actual test execution.",
        )
    # Analysis failed. Returning a generic response.
    return (
        None,
        "Please ensure the passed test files and failed test files in your test script are generated from actual test execution.",
    )


def _parse_legal_response(response: str) -> bool | None:
    response = response.strip()
    if not response:
        return None
    payload = None
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                payload = json.loads(response[start : end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(payload, dict):
        return None
    legal_value = payload.get("legal")
    if isinstance(legal_value, bool):
        return legal_value
    if isinstance(legal_value, str):
        normalized = legal_value.strip().lower()
        if normalized in {"true", "yes", "1", "correct", "legal", "true."}:
            return True
        if normalized in {"false", "no", "0", "incorrect", "illegal", "false."}:
            return False
    return None


def request_host_validation(
    *,
    dockerfile_path: str,
    test_script_path: str,
    extra_info_path: str,
    timeout_seconds: int | None = None,
) -> ValidationResult:
    try:
        dockerfile_text = Path(dockerfile_path).read_text()
        test_script_text = Path(test_script_path).read_text()
        extra_info_text = Path(extra_info_path).read_text()
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, f"Failed to read artifacts: {exc}")

    payload = {
        "dockerfile": dockerfile_text,
        "test_script": test_script_text,
        "extra_info": extra_info_text,
    }

    host_gateway_ip = os.getenv("HOST_GATEWAY_IP", "172.17.0.1")
    if timeout_seconds is None:
        timeout_seconds = int(os.getenv("VALIDATOR_TOOL_REQUEST_TIMEOUT", "1800"))
    port = int(os.getenv("VALIDATOR_PORT", "9090"))
    url = f"http://{host_gateway_ip}:{port}/validate"

    try:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        return ValidationResult(False, f"Host validation request failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, f"Invalid response from host: {exc}")

    return ValidationResult(
        bool(data.get("ok")),
        str(data.get("message", "")),
    )


def register_validator_tool() -> None:
    register_tool("validator", ValidatorTool.create)


# Ensure tool is registered when module is imported on the client side
register_validator_tool()
