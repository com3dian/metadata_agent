"""
Defines the state and data structures for the agent's workflow.
"""

from typing import Any, Dict, List, Optional, TypedDict


class Artifact(TypedDict):
    """
    Represents a piece of data (a "material" or "resource") in the workspace.
    """

    name: str  # The unique name for the artifact, e.g., "raw_table_list".
    content: Any  # The actual data content (a list, a dataframe, a string).
    content_type: str  # A hint about the data type, e.g., "list_of_strings".
    producer: str  # The task name that created this artifact.


class AgentState(TypedDict):
    """
    The central state of the application, passed between nodes in the graph.
    It contains all information about the current run.
    """

    # Input & File Check
    dataset_path: str
    file_type: Optional[str]

    # --- Planning and Execution ---
    plan: List[Dict]  # The plan is now a list of dictionaries from the Pydantic model.
    completed_steps: List[Dict]

    # --- Workspace for Intermediate Results ---
    workspace: Dict[
        str, Artifact
    ]  # The "scratchpad" for storing all generated artifacts.

    # --- Output and Errors ---
    final_metadata: Dict[
        str, Any
    ]  # The final, user-facing metadata, assembled at the end.
    error: Optional[str]

    # --- History and Debugging ---
    history: List[str]
