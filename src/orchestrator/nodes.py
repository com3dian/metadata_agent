"""
This file contains the functions (nodes) that make up the agent's graph.
"""

import logging
import os
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI

from . import prompts
from . import utils
from .schemas import Plan
from .state import AgentState, Artifact

# This would be properly initialized in your main app
PLAYER_REGISTRY = {}
LLM_CHAIN = None


def setup_chains():
    """A helper function to set up the LLM chain."""
    global LLM_CHAIN
    prompt = prompts.get_planning_prompt()
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    structured_llm = llm.with_structured_output(Plan)
    LLM_CHAIN = prompt | structured_llm


# Call setup function to initialize the chain when the module is loaded.
setup_chains()


def initial_analysis_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes the initial dataset path to determine file type."""
    logging.info("---EXECUTING NODE: initial_analysis---")
    path = state["dataset_path"]
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}

    _, extension = os.path.splitext(path)
    file_type = extension.lower().strip(".") or "unknown"

    history = state.get("history", []) + [
        f"Initial analysis: Detected file type '{file_type}'."
    ]
    return {"file_type": file_type, "history": history}


def plan_generation_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates and validates a multi-step plan using an LLM.
    If the plan's dataflow is invalid, it returns an error.
    """
    logging.info("---EXECUTING NODE: plan_generation---")
    file_type = state.get("file_type")
    if not file_type or file_type == "unknown":
        return {"error": "Cannot generate plan for unknown file type."}

    pydantic_plan = LLM_CHAIN.invoke({
        "file_type": file_type,
        "available_players": list(PLAYER_REGISTRY.keys()),
    })
    
    plan_as_dicts = [step.dict() for step in pydantic_plan.steps]

    # --- New Validation Step ---
    is_valid, message = utils.validate_plan_dataflow(plan_as_dicts)
    if not is_valid:
        error_message = f"Invalid Plan Generated: {message}"
        logging.error(error_message)
        # Stop execution by putting an error in the state
        return {"error": error_message}
    # --- End of Validation Step ---

    history = state.get("history", []) + [f"Generated and validated a {len(plan_as_dicts)}-step plan."]
    return {"plan": plan_as_dicts, "history": history}


def execute_step_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes a single step from the plan, acting as the primary "engine" of the agent.

    This node is responsible for taking a `PlanStep`, resolving its data dependencies
    from the workspace, executing the specified task using the correct player, and
    storing the results back into the workspace as new artifacts.

    Core Logic:
    1.  Pops the next `PlanStep` from the `plan` list in the state.
    2.  Resolves Input Dependencies: For each entry in the step's `inputs` map,
        it retrieves the corresponding `Artifact` from the `workspace`.
    3.  Prepares Arguments: It constructs a dictionary of keyword arguments (`kwargs`)
        to pass to the player's task, where the key is the parameter name from
        the `inputs` map and the value is the content of the resolved artifact.
    4.  Executes Task: It dynamically calls the `execute_task` method on the
        specified player, passing the prepared arguments using the `**kwargs` syntax.
    5.  Stores Outputs: It takes the return value from the task and creates a new
        `Artifact` for each name in the step's `outputs` list. These new
        artifacts are then saved into the `workspace` for future steps to use.

    State Updates:
    - `plan`: The executed step is removed from the front of the list.
    - `completed_steps`: The executed step is added to this list for logging.
    - `workspace`: New artifacts produced by the step are added.
    - `history`: A message is appended to log the successful execution.
    - `error`: This field is populated if any part of the process fails.

    Args:
        state (AgentState): The current state of the agent's workflow, containing
                            the plan and the workspace.

    Returns:
        Dict[str, Any]: A dictionary containing the updated state fields to be
                        merged back into the main agent state.
    """
    logging.info("---EXECUTING NODE: execute_step---")
    plan = state.get("plan", [])
    if not plan:
        return {"error": "Cannot execute step: plan is empty."}

    step = plan.pop(0)
    task_name = step["task"]
    player_name = step["player"]
    inputs = step["inputs"]
    outputs = step["outputs"]

    player = PLAYER_REGISTRY.get(player_name)
    if not player:
        return {"error": f"Player '{player_name}' not found."}

    try:
        # 1. Resolve inputs: Fetch required artifacts from the workspace
        kwargs = {}
        for param_name, artifact_name in inputs.items():
            if artifact_name not in state["workspace"]:
                return {
                    "error": f"Artifact '{artifact_name}' needed for task '{task_name}' not found in workspace."
                }
            kwargs[param_name] = state["workspace"][artifact_name]["content"]

        # 2. Execute the task
        logging.info(f"    Executing task '{task_name}' with player '{player_name}'.")
        result = player.execute_task(task_name, **kwargs)

        # 3. Store outputs in the workspace
        workspace = state["workspace"].copy()
        if len(outputs) == 1:
            # Single output case
            artifact_name = outputs[0]
            new_artifact = Artifact(
                name=artifact_name,
                content=result,
                content_type=type(result).__name__,
                producer=task_name,
            )
            workspace[artifact_name] = new_artifact
            logging.info(f"    Stored artifact '{artifact_name}' in workspace.")
        # NOTE: Add logic here for multiple outputs if a task can return a tuple/dict

        history = state.get("history", []) + [f"Successfully executed: '{task_name}'."]
        return {
            "plan": plan,
            "completed_steps": state.get("completed_steps", []) + [step],
            "workspace": workspace,
            "history": history,
        }

    except Exception as e:
        logging.error(f"Error executing task '{task_name}': {e}")
        return {"error": f"Error in task '{task_name}': {e}"}


def compile_results_node(state: AgentState) -> Dict[str, Any]:
    """Compiles the final metadata from all artifacts in the workspace."""
    logging.info("---EXECUTING NODE: compile_results---")

    final_metadata = {}
    for artifact_name, artifact in state["workspace"].items():
        # Simple merge: use artifact name as key. Could be more sophisticated.
        final_metadata[artifact_name] = artifact["content"]

    history = state.get("history", []) + [
        "Finished: Compiled final metadata from workspace."
    ]
    return {"final_metadata": final_metadata, "history": history}


def plan_router_node(state: AgentState) -> str:
    """Determines the next node to execute."""
    logging.info("---ROUTING: Checking plan status---")
    if state.get("error"):
        return "__end__"
    if state.get("plan"):
        return "execute_step_node"
    else:
        return "compile_results_node"
