"""
Textual-based terminal UI for Metadata Agent.
"""

from pathlib import Path
import logging
import queue
import threading

from rich.highlighter import Highlighter
from rich.text import Text
from textual.app import App, ComposeResult
from textual.events import Key
from textual.containers import Container, Vertical
from textual.widgets import Input, Static
from src.tui.palette import (
    COMMAND_NAME_COLOR,
    DESCRIPTION_COLOR,
    DIALOG_COLOR,
    GLOBAL_BACKGROUND,
    INPUT_BACKGROUND,
    INPUT_WARNING_COLOR,
    NO_MATCH_COMMAND_COLOR,
    REFERENCE_BACKGROUND,
    REFERENCE_TEXT_COLOR,
    STANDARD_REFERENCE_BACKGROUND,
    STANDARD_REFERENCE_TEXT_COLOR,
    SUGGESTION_COLOR,
    TITLE_COLOR,
)
from src.standards import METADATA_STANDARDS
from src.config import DEFAULT_TOPOLOGY, LLM_PROVIDER, PLANNING_TEMPERATURE, get_model_name
from src.tui.reference_session import ReferenceSessionState
from src.tui.references import (
    extract_invalid_standard_mentions,
    extract_standard_mentions,
    get_at_suggestions,
    stylize_verified_references,
)
from src.orchestrator import Orchestrator


class VerifiedReferenceHighlighter(Highlighter):
    """Highlight verified @path references in input text."""

    def __init__(self, app: "MetadataTUI") -> None:
        super().__init__()
        self.app = app

    def highlight(self, text: Text) -> None:
        self.app._stylize_verified_references(text)


class TUILogHandler(logging.Handler):
    """Forward Python logging records into the TUI event queue."""

    def __init__(self, app: "MetadataTUI") -> None:
        super().__init__(level=logging.INFO)
        self.app = app

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self.app.ui_events.put(("log", message, False))
        except Exception:
            # Logging should never crash the UI.
            pass


class MetadataTUI(App):
    """Minimal one-request, one-answer Textual app."""
    EXIT_COMMAND_LABEL = "/quit; /exit; /bye"
    SLASH_COMMANDS = {
        "/help": "Show available commands",
        "/clear": "Clear dialog history",
        EXIT_COMMAND_LABEL: "Exit the app",
    }
    EXIT_COMMANDS = {"/quit", "/exit", "/bye"}

    CSS = (
        """
    Screen {
        padding: 0;
        background: __GLOBAL_BACKGROUND__;
    }

    #panel {
        width: 100%;
        height: 1fr;
        padding: 0;
        layout: vertical;
        overflow-y: auto;
    }

    #feed {
        width: 100%;
        height: auto;
    }

    #title {
        text-style: bold;
        color: __TITLE_COLOR__;
        margin: 0 0 1 0;
    }

    #welcome {
        color: __DESCRIPTION_COLOR__;
        margin: 0 0 1 0;
    }

    #user_input {
        height: 3;
        margin: 0;
        padding: 1 2;
        border: none;
        background: __INPUT_BACKGROUND__;
    }

    #command_suggestions {
        margin: 0 2 1 2;
        color: __SUGGESTION_COLOR__;
    }

    #dialog {
        color: __DIALOG_COLOR__;
    }

    #messages {
        width: 100%;
        layout: vertical;
        height: auto;
    }

    .message {
        width: 100%;
        margin: 0;
    }

    .user-message {
        background: __INPUT_BACKGROUND__;
        padding: 1 2;
        align: center middle;
    }

    .assistant-message {
        padding: 1 2;
        align: center middle;
    }
    """
        .replace("__TITLE_COLOR__", TITLE_COLOR)
        .replace("__DESCRIPTION_COLOR__", DESCRIPTION_COLOR)
        .replace("__INPUT_BACKGROUND__", INPUT_BACKGROUND)
        .replace("__SUGGESTION_COLOR__", SUGGESTION_COLOR)
        .replace("__DIALOG_COLOR__", DIALOG_COLOR)
        .replace("__GLOBAL_BACKGROUND__", GLOBAL_BACKGROUND)
    )

    BINDINGS = [("ctrl+c", "quit", "Quit"), ("escape", "quit", "Quit")]
    reference_root = Path.cwd()
    standard_names = sorted(METADATA_STANDARDS.keys())
    reference_session: ReferenceSessionState
    orchestrator: Orchestrator | None = None
    backend_running: bool = False
    ui_events: "queue.Queue[tuple[str, str, bool]]"
    tui_log_handler: TUILogHandler | None = None

    @staticmethod
    def _get_active_token(text: str, cursor_position: int) -> str:
        prefix = text[:cursor_position]
        if not prefix:
            return ""
        token_start = max(prefix.rfind(" "), prefix.rfind("\t"), prefix.rfind("\n")) + 1
        return prefix[token_start:]

    def _stylize_verified_references(self, rendered: Text) -> Text:
        return stylize_verified_references(
            rendered,
            root=self.reference_root,
            standard_names=self.standard_names,
            text_color=REFERENCE_TEXT_COLOR,
            background_color=REFERENCE_BACKGROUND,
            standard_text_color=STANDARD_REFERENCE_TEXT_COLOR,
            standard_background_color=STANDARD_REFERENCE_BACKGROUND,
        )

    def _render_text_with_verified_references(self, text: str, *, markup: bool = False) -> Text:
        rendered = Text.from_markup(text) if markup else Text(text)
        return self._stylize_verified_references(rendered)

    def on_mount(self) -> None:
        self.ui_events = queue.Queue()
        self.set_interval(0.1, self._flush_ui_events)

        self.reference_session = ReferenceSessionState()
        try:
            self.orchestrator = Orchestrator(
                topology_name=DEFAULT_TOPOLOGY,
                model_name=get_model_name(),
                temperature=PLANNING_TEMPERATURE,
                provider=LLM_PROVIDER,
            )
        except Exception:
            # Keep TUI usable even if backend initialization fails.
            self.orchestrator = None

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        self.tui_log_handler = TUILogHandler(self)
        self.tui_log_handler.setFormatter(
            logging.Formatter("[log] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        )
        root_logger.addHandler(self.tui_log_handler)

        self.query_one("#user_input", Input).focus()
        self.query_one("#panel", Container).scroll_home(animate=False)

    def _flush_ui_events(self) -> None:
        while True:
            try:
                kind, message, markup = self.ui_events.get_nowait()
            except queue.Empty:
                break

            if kind == "message":
                self._append_dialog(message, markup=markup)
            elif kind == "done":
                self.backend_running = False
                input_widget = self.query_one("#user_input", Input)
                suggestions_widget = self.query_one("#command_suggestions", Static)
                input_widget.disabled = False
                input_widget.display = True
                suggestions_widget.display = True
            elif kind == "log":
                self._append_dialog(message)

    def _run_pipeline_worker(
        self,
        *,
        source_files: list[str],
        chosen_standard: str,
    ) -> None:
        if self.orchestrator is None:
            self.ui_events.put(("message", "[red]Backend is not initialized. Unable to run pipeline.[/red]", True))
            self.ui_events.put(("done", "", False))
            return

        try:
            standard_prompt = METADATA_STANDARDS[chosen_standard]
            result = self.orchestrator.run(
                source=source_files,
                metadata_standard=standard_prompt,
                metadata_standard_name=chosen_standard,
                name="tui_context",
            )
            if result is None:
                self.ui_events.put(("message", "[red]Pipeline run failed: no result returned.[/red]", True))
            elif result.final_metadata:
                self.ui_events.put(("message", f"Pipeline completed with @{chosen_standard}.", False))
                self.ui_events.put(("message", str(result.final_metadata), False))
            else:
                self.ui_events.put(
                    (
                        "message",
                        f"Pipeline completed with @{chosen_standard}, but no final metadata was produced.",
                        False,
                    )
                )
        except Exception as exc:
            self.ui_events.put(("message", f"[red]Pipeline run failed: {exc}[/red]", True))
        finally:
            self.ui_events.put(("done", "", False))

    def compose(self) -> ComposeResult:
        yield Container(
            Vertical(
                Static("Metadata Agent", id="title"),
                Static(
                    "This tool works with dataset and table metadata. "
                    "Describe your need, or type /help for available commands.",
                    id="welcome",
                ),
                Vertical(id="messages"),
                Input(
                    placeholder="Type your request (or /help) and press Enter...",
                    highlighter=VerifiedReferenceHighlighter(self),
                    id="user_input",
                ),
                Static("", id="command_suggestions"),
                id="feed",
            ),
            id="panel",
        )

    def _append_dialog(self, text: str, *, is_user: bool = False, markup: bool = False) -> None:
        messages = self.query_one("#messages", Vertical)
        classes = "message user-message" if is_user else "message assistant-message"
        messages.mount(
            Static(
                self._render_text_with_verified_references(text, markup=markup),
                classes=classes,
            )
        )
        self.query_one("#panel", Container).scroll_end(animate=False)

    def _update_suggestions(self, current_text: str, cursor_position: int) -> None:
        suggestions_widget = self.query_one("#command_suggestions", Static)
        warning_text = ""
        standard_mentions = extract_standard_mentions(current_text, self.standard_names)
        if len(standard_mentions) > 1:
            warning_text = (
                f"[{INPUT_WARNING_COLOR}]You should only provide one standard.[/{INPUT_WARNING_COLOR}]"
            )

        active_token = self._get_active_token(current_text, cursor_position)
        if not active_token:
            suggestions_widget.update(warning_text)
            return

        if active_token.startswith("/") and current_text[:cursor_position].lstrip() == active_token:
            matches = [
                (command, description)
                for command, description in self.SLASH_COMMANDS.items()
                if (
                    command.startswith(active_token)
                    or (
                        command == self.EXIT_COMMAND_LABEL
                        and any(exit_cmd.startswith(active_token) for exit_cmd in self.EXIT_COMMANDS)
                    )
                )
            ]

            if not matches:
                suggestion_text = (
                    f"[{NO_MATCH_COMMAND_COLOR}]No matching commands[/{NO_MATCH_COMMAND_COLOR}]"
                )
                suggestions_widget.update(
                    f"{warning_text}\n{suggestion_text}".strip() if warning_text else suggestion_text
                )
                return

            suggestions_text = "\n".join(
                f"[bold {COMMAND_NAME_COLOR}]{command}[/bold {COMMAND_NAME_COLOR}] - {description}"
                for command, description in matches
            )
            suggestions_widget.update(
                f"{warning_text}\n{suggestions_text}".strip() if warning_text else suggestions_text
            )
            return

        if active_token.startswith("@"):
            suggestions = get_at_suggestions(
                active_token,
                root=self.reference_root,
                standard_names=self.standard_names,
            )

            if not suggestions:
                suggestion_text = (
                    f"[{NO_MATCH_COMMAND_COLOR}]No matching files or standards[/{NO_MATCH_COMMAND_COLOR}]"
                )
                suggestions_widget.update(
                    f"{warning_text}\n{suggestion_text}".strip() if warning_text else suggestion_text
                )
                return

            suggestions_text = "\n".join(
                (
                    f"[{STANDARD_REFERENCE_TEXT_COLOR} on {STANDARD_REFERENCE_BACKGROUND}]@{item.token}[/{STANDARD_REFERENCE_TEXT_COLOR} on {STANDARD_REFERENCE_BACKGROUND}]"
                    if item.kind == "standard"
                    else f"[{REFERENCE_TEXT_COLOR} on {REFERENCE_BACKGROUND}]@{item.token}[/{REFERENCE_TEXT_COLOR} on {REFERENCE_BACKGROUND}]"
                )
                for item in suggestions
            )
            suggestions_widget.update(
                f"{warning_text}\n{suggestions_text}".strip() if warning_text else suggestions_text
            )
            return

        suggestions_widget.update(warning_text)

    def _handle_command(self, command_text: str) -> None:
        if command_text == "/help":
            self._append_dialog(
                "[bold cyan]Available commands[/bold cyan]\n"
                "- /help: Show available commands\n"
                "- /clear: Clear dialog history\n"
                "- /quit (/exit, /bye): Exit the app",
                markup=True,
            )
            return
        if command_text == "/clear":
            messages = self.query_one("#messages", Vertical)
            messages.remove_children()
            self.reference_session.clear_current()
            return
        if command_text in self.EXIT_COMMANDS:
            self.exit()
            return
        self._append_dialog(
            f"[yellow]Unknown command:[/yellow] {command_text}",
            markup=True,
        )

    def on_input_submitted(self, _: Input.Submitted) -> None:
        if self.backend_running:
            self._append_dialog(
                "[red]Pipeline is already running. Please wait for completion.[/red]",
                markup=True,
            )
            return

        input_widget = self.query_one("#user_input", Input)
        user_text = input_widget.value.strip()
        input_widget.value = ""
        self._update_suggestions("", 0)
        self.reference_session.clear_current()

        if not user_text:
            return

        if user_text.startswith("/"):
            self._handle_command(user_text)
            return

        referenced_items = self.reference_session.commit_current_input(
            user_text,
            self.reference_root,
        )
        referenced_files = [ref for ref in referenced_items if ref.is_file]
        referenced_standards = extract_standard_mentions(user_text, self.standard_names)
        invalid_standards = extract_invalid_standard_mentions(
            user_text,
            standard_names=self.standard_names,
            root=self.reference_root,
        )

        self._append_dialog(user_text, is_user=True)

        if invalid_standards:
            self._append_dialog(
                (
                    f"[red]Invalid @standard reference(s): {', '.join(f'@{s}' for s in invalid_standards)}. "
                    "Please use a valid standard name.[/red]"
                ),
                markup=True,
            )
            return

        if not referenced_files:
            self._append_dialog(
                "[red]Please include at least one valid @file reference to run the backend.[/red]",
                markup=True,
            )
            return

        if referenced_standards:
            chosen_standard = referenced_standards[0]
            if len(referenced_standards) > 1:
                self._append_dialog(
                    (
                        f"[red]Multiple standards detected ({', '.join(referenced_standards)}). "
                        f"Using @{chosen_standard}.[/red]"
                    ),
                    markup=True,
                )

            source = [str(ref.absolute_path) for ref in referenced_files]
            self.backend_running = True
            input_widget.disabled = True
            input_widget.display = False
            self.query_one("#command_suggestions", Static).display = False
            self._append_dialog(f"Running pipeline with @{chosen_standard}...")
            threading.Thread(
                target=self._run_pipeline_worker,
                kwargs={"source_files": source, "chosen_standard": chosen_standard},
                daemon=True,
            ).start()
            return

        # No standard provided yet: return a dummy confirmation for now.
        file_summary = ", ".join(ref.path_name for ref in referenced_files)
        self._append_dialog(
            f"Captured valid file context: {file_summary}. No @standard provided yet.",
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        self.reference_session.update_from_input(event.value, self.reference_root)
        self._update_suggestions(event.value, event.input.cursor_position)

    def on_key(self, event: Key) -> None:
        if event.key != "tab":
            return

        input_widget = self.query_one("#user_input", Input)
        if not input_widget.has_focus:
            return

        current_text = input_widget.value
        cursor_position = input_widget.cursor_position
        active_token = self._get_active_token(current_text, cursor_position)
        suggestions = get_at_suggestions(
            active_token,
            root=self.reference_root,
            standard_names=self.standard_names,
        )
        matches = [item.token for item in suggestions]

        if len(matches) != 1:
            return

        token_start = cursor_position - len(active_token)
        completed_token = f"@{matches[0]}"
        input_widget.value = (
            f"{current_text[:token_start]}{completed_token}{current_text[cursor_position:]}"
        )
        input_widget.cursor_position = token_start + len(completed_token)
        self._update_suggestions(input_widget.value, input_widget.cursor_position)
        event.prevent_default()
        event.stop()


def run_tui() -> None:
    """Run the Textual TUI app."""
    MetadataTUI().run()
