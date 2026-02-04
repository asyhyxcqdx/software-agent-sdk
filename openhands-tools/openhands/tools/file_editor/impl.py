from pathlib import Path
from typing import TYPE_CHECKING

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import ToolExecutor


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
from openhands.tools.file_editor.definition import (
    CommandLiteral,
    FileEditorAction,
    FileEditorObservation,
)
from openhands.tools.file_editor.editor import FileEditor
from openhands.tools.file_editor.exceptions import ToolError
from openhands.tools.utils.run_tests_guard import REMINDER_TEXT, record_run_tests_edit


# Module-global editor instance (lazily initialized in file_editor)
_GLOBAL_EDITOR: FileEditor | None = None


class FileEditorExecutor(ToolExecutor):
    """File editor executor with configurable file restrictions."""

    def __init__(
        self,
        workspace_root: str | None = None,
        allowed_edits_files: list[str] | None = None,
    ):
        self.editor: FileEditor = FileEditor(workspace_root=workspace_root)
        self.allowed_edits_files: set[Path] | None = (
            {Path(f).resolve() for f in allowed_edits_files}
            if allowed_edits_files
            else None
        )

    def __call__(
        self,
        action: FileEditorAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> FileEditorObservation:
        # Enforce allowed_edits_files restrictions
        if self.allowed_edits_files is not None and action.command != "view":
            action_path = Path(action.path).resolve()
            if action_path not in self.allowed_edits_files:
                return FileEditorObservation.from_text(
                    text=(
                        f"Operation '{action.command}' is not allowed "
                        f"on file '{action_path}'. "
                        f"Only the following files can be edited: "
                        f"{sorted(str(p) for p in self.allowed_edits_files)}"
                    ),
                    command=action.command,
                    is_error=True,
                )

        result: FileEditorObservation | None = None
        try:
            result = self.editor(
                command=action.command,
                path=action.path,
                file_text=action.file_text,
                view_range=action.view_range,
                old_str=action.old_str,
                new_str=action.new_str,
                insert_line=action.insert_line,
            )
        except ToolError as e:
            result = FileEditorObservation.from_text(
                text=e.message, command=action.command, is_error=True
            )
        assert result is not None, "file_editor should always return a result"
        if (
            not result.is_error
            and action.command in {"create", "str_replace", "insert", "undo_edit"}
        ):
            try:
                edited_path = Path(result.path or action.path).resolve()
            except Exception:
                edited_path = None
            if edited_path and record_run_tests_edit(edited_path, conversation):
                reminder = TextContent(text=f"\n\n{REMINDER_TEXT}")
                result = result.model_copy(
                    update={"content": list(result.content) + [reminder]}
                )
        return result


def file_editor(
    command: CommandLiteral,
    path: str,
    file_text: str | None = None,
    view_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
) -> FileEditorObservation:
    """A global FileEditor instance to be used by the tool."""

    global _GLOBAL_EDITOR
    if _GLOBAL_EDITOR is None:
        _GLOBAL_EDITOR = FileEditor()

    result: FileEditorObservation | None = None
    try:
        result = _GLOBAL_EDITOR(
            command=command,
            path=path,
            file_text=file_text,
            view_range=view_range,
            old_str=old_str,
            new_str=new_str,
            insert_line=insert_line,
        )
    except ToolError as e:
        result = FileEditorObservation.from_text(
            text=e.message, command=command, is_error=True
        )
    assert result is not None, "file_editor should always return a result"
    return result
