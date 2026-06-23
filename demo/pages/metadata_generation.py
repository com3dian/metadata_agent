"""Streamlit page for the metadata generation demo.

The page owns only UI concerns: collecting user inputs, showing a data preview,
triggering the workflow, caching results in Streamlit session state, and
rendering the workflow output.
"""

import multiprocessing
from queue import Empty
from typing import Any

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from demo.workflows.metadata_generation import (
    SUPPORTED_FILE_TYPES,
    available_metadata_standards,
    execution_details,
    extract_metadata,
    generate_metadata,
    load_preview,
    uploaded_file_key,
)
from src.standards import load_metadata_standard


def render_controls() -> tuple[UploadedFile | None, str]:
    """Render upload and standard selection controls.

    Returns:
        The uploaded file, if present, and the selected metadata standard name.
    """
    uploaded_file = st.file_uploader(
        "Upload a dataset: ",
        type=SUPPORTED_FILE_TYPES,
        help="CSV and TSV files are supported by the demo context.",
    )
    standard_name = st.selectbox(
        "Choose a metadata standard: ",
        options=available_metadata_standards(),
        index=0,
    )
    st.caption("Metadata standard preview")
    st.code(load_metadata_standard(standard_name), language="json")
    return uploaded_file, standard_name


def render_preview(uploaded_file: UploadedFile, file_bytes: bytes) -> None:
    """Render a preview table for the uploaded dataset.

    Args:
        uploaded_file: Streamlit uploaded file object.
        file_bytes: Raw uploaded file bytes.
    """
    st.subheader("Data preview")
    try:
        st.dataframe(
            load_preview(uploaded_file.name, file_bytes),
            width='stretch',
        )
        st.caption(f"{uploaded_file.name} - {len(file_bytes):,} bytes")
    except Exception as exc:
        st.warning(f"Preview unavailable: {exc}")


def run_generation(
    file_name: str,
    file_bytes: bytes,
    standard_name: str,
    messages: multiprocessing.Queue,
) -> None:
    """Run metadata generation in a child process and publish its output.

    Args:
        file_name: Name of the uploaded file.
        file_bytes: Raw uploaded file bytes.
        standard_name: Selected metadata standard name.
        messages: Queue used to publish progress and the final result.
    """
    def show_progress(stage: str, payload: Any | None) -> None:
        messages.put(("progress", stage, payload))

    try:
        result = generate_metadata(
            file_name,
            file_bytes,
            standard_name,
            progress_callback=show_progress,
        )
        messages.put(("result", result))
    except Exception as exc:
        messages.put(("error", repr(exc)))


def stop_generation(job: dict[str, Any]) -> None:
    """Terminate an active metadata generation process."""
    process = job["process"]
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)


def render_result(result: dict[str, Any]) -> None:
    """Render generated metadata and compact execution details.

    Args:
        result: JSON-friendly metadata generation result.
    """
    metadata = extract_metadata(result)

    st.subheader("Generated metadata")
    if metadata:
        st.json(metadata)
    else:
        st.warning("No final metadata artifact was found in the result.")

    plan = result.get("generated_plan") or []
    if plan:
        with st.expander("Generated plan", expanded=True):
            st.json(plan)

    step_results = result.get("step_results") or []
    if step_results:
        st.subheader("Execution steps")
        st.dataframe(
            [
                {
                    "step": step.get("step_index", 0) + 1,
                    "task": step.get("task"),
                    "player": step.get("player_role"),
                    "success": step.get("success"),
                    "artifacts": ", ".join((step.get("artifacts") or {}).keys()),
                    "error": step.get("error"),
                }
                for step in step_results
            ],
            width='stretch',
            hide_index=True,
        )

    with st.expander("Execution details"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Plan steps", result.get("plan_steps_count", 0))
        col2.metric("Completed", result.get("steps_completed", 0))
        col3.metric("Success", "Yes" if result.get("success") else "No")

        if result.get("error"):
            st.error(result["error"])

        st.json(execution_details(result))


def main() -> None:
    """Render the metadata generation Streamlit page."""
    st.set_page_config(page_title="Metadata Generation", page_icon="MD", layout="wide")
    # st.title("Metadata Generation")

    controls_col, preview_col = st.columns([1, 2], gap="large")

    with controls_col:
        uploaded_file, standard_name = render_controls()

    if uploaded_file is None:
        with preview_col:
            st.info("Upload a CSV or TSV file to start.")
        st.stop()

    file_bytes = uploaded_file.getvalue()
    file_key = uploaded_file_key(file_bytes, standard_name)

    with preview_col:
        render_preview(uploaded_file, file_bytes)

    if "metadata_generation_results" not in st.session_state:
        st.session_state.metadata_generation_results = {}

    @st.fragment(run_every=0.5)
    def generation_controls() -> None:
        action_col, output_col = st.columns([1, 2], gap="large")
        job = st.session_state.get("metadata_generation_job")
        if job is not None and job["file_key"] != file_key:
            stop_generation(job)
            st.session_state.metadata_generation_job = None
            job = None

        running = job is not None and job["process"].is_alive()
        with action_col:
            clicked = st.button(
                "Stop generation" if running else "Generate metadata",
                type="primary",
                width='stretch',
            )

        if clicked and running:
            stop_generation(job)
            st.session_state.metadata_generation_job = None
            with output_col:
                st.warning("Metadata generation stopped.")
            st.rerun(scope="fragment")

        if clicked and not running:
            context = multiprocessing.get_context("spawn")
            messages = context.Queue()
            process = context.Process(
                target=run_generation,
                args=(uploaded_file.name, file_bytes, standard_name, messages),
                daemon=True,
            )
            process.start()
            job = {
                "file_key": file_key,
                "process": process,
                "messages": messages,
                "progress": [],
            }
            st.session_state.metadata_generation_job = job
            running = True

        if job is not None:
            while True:
                try:
                    message = job["messages"].get_nowait()
                except Empty:
                    break
                if message[0] == "progress":
                    job["progress"].append(message[1:])
                elif message[0] == "result":
                    st.session_state.metadata_generation_results[file_key] = message[1]
                    job["process"].join(timeout=1)
                    st.session_state.metadata_generation_job = None
                    with output_col:
                        render_result(message[1])
                    return
                elif message[0] == "error":
                    st.session_state.metadata_generation_job = None
                    with output_col:
                        st.error(f"Metadata generation failed: {message[1]}")
                    return

            with output_col:
                status = st.status(
                    "Generating metadata",
                    expanded=True,
                    state="running",
                )
                status.write("File uploaded and staged for analysis.")
                for stage, payload in job["progress"]:
                    if stage == "context_created":
                        status.write("Execution context created.")
                        with status.expander("Context"):
                            st.json(payload)
                    elif stage == "plan_generated":
                        status.write("Execution plan generated.")
                        with status.expander("Generated plan"):
                            st.json(payload)
                    elif stage == "execution_complete":
                        status.write("Plan execution complete.")
                if not job["process"].is_alive():
                    st.session_state.metadata_generation_job = None
                    st.error("Metadata generation stopped unexpectedly.")
            return

        with output_col:
            result = st.session_state.metadata_generation_results.get(file_key)
            if result is None:
                st.caption("Click Generate metadata to run the agent.")
            else:
                render_result(result)

    generation_controls()


if __name__ == "__main__":
    main()
