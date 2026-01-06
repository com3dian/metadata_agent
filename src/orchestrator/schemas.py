"""
Defines the Pydantic models for structured LLM output.
This ensures the planner's output is reliable and easy to work with.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class PlanStep(BaseModel):
    """
    Defines a single, executable step in the agent's plan, including data dependencies.
    """
    task: str = Field(description="The specific and single task to be performed, e.g., 'get_row_count'.")
    player: str = Field(description="The name of the player responsible for executing this task, e.g., 'tabular_player'.")
    rationale: str = Field(description="The reasoning for why this step is necessary.")
    
    # Defines the dataflow dependencies for this step
    inputs: Dict[str, str] = Field(default_factory=dict, description="Maps a task's parameter names to the names of artifacts in the workspace that should be used as input.")
    outputs: List[str] = Field(default_factory=list, description="A list of new artifact names that this step will produce and save to the workspace.")

class Plan(BaseModel):
    """The complete, multi-step plan for the agent to execute."""
    steps: List[PlanStep] = Field(description="The list of sequential steps the agent should follow to extract metadata.")