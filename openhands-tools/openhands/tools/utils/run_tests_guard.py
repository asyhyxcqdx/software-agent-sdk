from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


_TARGET_PATH = Path("/output/run_tests.py").resolve()
REMINDER_TEXT = (
    "Reminder: /output/run_tests.py has been modified 3 times. "
    "FOCUS ON THE PRIMARY GOAL: get a runnable environment, make sure "
    "run_tests.py EXITS NORMALLY, and ensure 'passed test files' contains a "
    "reasonable number of unit tests. That's enough. "
    "DO NOT get stuck on a small number of failed test files or their root causes. "
    "It is okay to leave the failed test list as-is."
)


@dataclass
class _RunTestsEditState:
    count: int = 0


_STATE_BY_CONVERSATION: dict[str, _RunTestsEditState] = {}


def _conversation_key(conversation: LocalConversation | None) -> str:
    if conversation is None:
        return "global"
    try:
        return str(conversation.state.id)
    except Exception:
        return "global"


def _get_state(conversation: LocalConversation | None) -> _RunTestsEditState:
    key = _conversation_key(conversation)
    state = _STATE_BY_CONVERSATION.get(key)
    if state is None:
        state = _RunTestsEditState()
        _STATE_BY_CONVERSATION[key] = state
    return state


def _maybe_remind(state: _RunTestsEditState) -> bool:
    if state.count >= 3:
        state.count = 0
        return True
    return False


def record_run_tests_edit(
    path: Path, conversation: LocalConversation | None = None
) -> bool:
    """Record a direct edit to /output/run_tests.py and return if reminder is due."""
    if path.resolve() != _TARGET_PATH:
        return False
    state = _get_state(conversation)
    state.count += 1
    return _maybe_remind(state)
