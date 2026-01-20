"""
Defines the Pydantic models for structured LLM output.
This ensures the planner's output is reliable and easy to work with.

Note: DataSource configuration and relationship discovery are now handled
by the unified DataSource abstraction in src/datasource/. This module
focuses on planning and execution result schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ===================================================================
#  PLAN STEP MODELS
# ===================================================================

class PlanStep(BaseModel):
    """
    Defines a single, executable step in the agent's plan, including data dependencies.
    
    Each step represents a task that will be executed by one or more players
    in parallel, followed by a debate phase to consolidate results.
    """
    task: str = Field(
        description="The specific and single task to be performed, e.g., 'get_row_count'."
    )
    player: str = Field(
        description="The name of the player role responsible for executing this task, e.g., 'data_analyst'."
    )
    rationale: str = Field(
        description="The reasoning for why this step is necessary."
    )
    
    # Target tables for multi-file datasets
    target_tables: List[str] = Field(
        default_factory=list,
        description="List of table names this step should operate on. Empty means all tables or dataset-level operation."
    )
    
    # Defines the dataflow dependencies for this step
    inputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps a task's parameter names to the names of artifacts in the workspace that should be used as input."
    )
    outputs: List[str] = Field(
        default_factory=list,
        description="A list of new artifact names that this step will produce and save to the workspace."
    )


class Plan(BaseModel):
    """The complete, multi-step plan for the agent to execute."""
    steps: List[PlanStep] = Field(
        description="The list of sequential steps the agent should follow to extract metadata."
    )
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert plan steps to a list of dictionaries for state management."""
        return [step.model_dump() for step in self.steps]


class StepResult(BaseModel):
    """
    The result of executing a single plan step.
    """
    step_index: int = Field(description="The index of the step in the plan.")
    task: str = Field(description="The task that was executed.")
    player_role: str = Field(description="The player role that executed this step.")
    
    # Results from parallel execution
    individual_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from each player that worked on this step."
    )
    
    # Debate summary
    debate_rounds_completed: int = Field(
        default=0,
        description="Number of debate rounds that were completed."
    )
    
    # Final consolidated result
    consolidated_result: str = Field(
        default="",
        description="The synthesized result after debate."
    )
    
    # Produced artifacts
    artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Artifacts produced by this step to add to workspace."
    )
    
    success: bool = Field(
        default=True,
        description="Whether the step completed successfully."
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the step failed."
    )


class ExecutionResult(BaseModel):
    """
    The complete result of executing a plan.
    """
    plan_steps_count: int = Field(description="Total number of steps in the plan.")
    steps_completed: int = Field(description="Number of steps successfully completed.")
    
    step_results: List[StepResult] = Field(
        default_factory=list,
        description="Results from each step."
    )
    
    final_workspace: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final state of the workspace with all artifacts."
    )
    
    final_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The final extracted metadata."
    )
    
    # DataSource results
    datasource_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information about the DataSource used."
    )
    
    table_metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-table metadata, keyed by table name."
    )
    
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relationships between tables (from DataSource)."
    )
    
    success: bool = Field(
        default=True,
        description="Whether the entire plan executed successfully."
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed."
    )
