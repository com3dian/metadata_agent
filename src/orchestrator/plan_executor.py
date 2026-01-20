"""
Plan Executor - Orchestrates the execution of a complete plan.

This module provides the PlanExecutor class that:
1. Takes a generated plan and execution topology
2. Iterates through each step sequentially
3. For each step, spawns parallel players and runs debates
4. Accumulates artifacts in a workspace
5. Produces the final metadata output

Uses the unified DataSource abstraction for all data access.
"""
import logging
from typing import Dict, Any, List, Optional

from .schemas import Plan, StepResult, ExecutionResult
from .step_executor import get_step_execution_graph, create_step_state
from ..datasource import DataSource
from ..topology import EXECUTION_TOPOLOGIES
from ..players import PLAYER_CONFIGS


class PlanExecutor:
    """
    Executes a complete plan using the specified topology.
    
    The executor iterates through each step in the plan, spawning
    parallel players and running debates as configured by the topology.
    
    Uses DataSource for unified data access across all source types.
    """
    
    def __init__(self, topology_name: str = "default"):
        """
        Initialize the PlanExecutor with a topology.
        
        Args:
            topology_name: Name of the execution topology to use
        """
        if topology_name not in EXECUTION_TOPOLOGIES:
            available = list(EXECUTION_TOPOLOGIES.keys())
            raise ValueError(
                f"Unknown topology '{topology_name}'. Available: {available}"
            )
        
        self.topology_name = topology_name
        self.topology = EXECUTION_TOPOLOGIES[topology_name]
        self.step_graph = get_step_execution_graph()
        
        logging.info(f"PlanExecutor initialized with topology: {topology_name}")
        logging.info(f"  Players per step: {self.topology['players_per_step']}")
        logging.info(f"  Debate rounds: {self.topology['debate_rounds']}")
        logging.info(f"  Player pool: {self.topology['player_pool']}")
    
    def execute(
        self,
        plan: Plan,
        datasource: DataSource,
        datasource_key: str,
        metadata_standard: str,
        player_pool: List[str] = None
    ) -> ExecutionResult:
        """
        Execute the complete plan.
        
        Args:
            plan: The Plan object with steps to execute
            datasource: The DataSource to analyze
            datasource_key: Key for the DataSource in the tool registry
            metadata_standard: The metadata standard to follow
            player_pool: Optional override for player pool (defaults to topology's pool)
            
        Returns:
            ExecutionResult with all step results and final metadata
        """
        # Use provided player_pool or fall back to topology's default
        effective_player_pool = player_pool or self.topology["player_pool"]
        logging.info("=" * 60)
        logging.info("STARTING PLAN EXECUTION")
        logging.info(f"Dataset: {datasource.name}")
        logging.info(f"Type: {datasource.source_type.value}")
        logging.info(f"Tables: {datasource.tables}")
        logging.info(f"Steps: {len(plan.steps)}")
        logging.info("=" * 60)
        
        # Initialize execution state
        workspace: Dict[str, Any] = {
            "_datasource_key": datasource_key,
            "_datasource_info": datasource.to_dict()
        }
        step_results: List[StepResult] = []
        table_metadata: Dict[str, Any] = {}
        
        # Pre-populate with schema info
        try:
            schema = datasource.get_schema()
            workspace["_schema"] = schema
        except Exception as e:
            logging.warning(f"Could not pre-load schema: {e}")
        
        # Convert plan steps to dict for processing
        plan_steps = plan.to_dict_list()
        
        # Execute each step
        for step_index, step_dict in enumerate(plan_steps):
            target_tables = step_dict.get("target_tables", [])
            
            logging.info("")
            logging.info(f"{'='*20} STEP {step_index + 1}/{len(plan_steps)} {'='*20}")
            logging.info(f"Task: {step_dict.get('task', 'Unknown')}")
            logging.info(f"Player: {step_dict.get('player', 'Unknown')}")
            logging.info(f"Rationale: {step_dict.get('rationale', 'None')}")
            if target_tables:
                logging.info(f"Target tables: {target_tables}")
            elif datasource.is_multi_table:
                logging.info(f"Target tables: ALL (dataset-level)")
            
            try:
                # Create step state with DataSource
                step_state = create_step_state(
                    step_index=step_index,
                    step_dict=step_dict,
                    datasource=datasource,
                    datasource_key=datasource_key,
                    workspace=workspace.copy(),
                    metadata_standard=metadata_standard,
                    players_per_step=self.topology["players_per_step"],
                    debate_rounds=self.topology["debate_rounds"],
                    player_pool=effective_player_pool
                )
                
                # Execute the step graph
                final_step_state = self.step_graph.invoke(step_state)
                
                # Check for errors
                if final_step_state.get("error"):
                    error_msg = final_step_state["error"]
                    logging.error(f"Step {step_index + 1} failed: {error_msg}")
                    
                    step_results.append(StepResult(
                        step_index=step_index,
                        task=step_dict.get("task", ""),
                        player_role=step_dict.get("player", ""),
                        individual_results=[],
                        debate_rounds_completed=final_step_state.get("current_debate_round", 0),
                        consolidated_result="",
                        artifacts={},
                        success=False,
                        error=error_msg
                    ))
                    
                    # Continue to next step (or could choose to abort)
                    continue
                
                # Extract results
                produced_artifacts = final_step_state.get("produced_artifacts", {})
                consolidated_result = final_step_state.get("consolidated_result", "")
                
                # Update workspace with new artifacts
                workspace.update(produced_artifacts)
                
                # Collect per-table metadata
                if datasource.is_multi_table and target_tables:
                    for table in target_tables:
                        if table not in table_metadata:
                            table_metadata[table] = {}
                        # Store table-specific artifacts
                        for artifact_name, artifact_value in produced_artifacts.items():
                            if artifact_name.startswith(f"{table}:"):
                                table_metadata[table][artifact_name] = artifact_value
                
                # Record step result
                step_results.append(StepResult(
                    step_index=step_index,
                    task=step_dict.get("task", ""),
                    player_role=step_dict.get("player", ""),
                    individual_results=final_step_state.get("player_results", []),
                    debate_rounds_completed=final_step_state.get("current_debate_round", 0),
                    consolidated_result=consolidated_result,
                    artifacts=produced_artifacts,
                    success=True
                ))
                
                logging.info(f"Step {step_index + 1} completed successfully")
                logging.info(f"  Artifacts produced: {list(produced_artifacts.keys())}")
                
            except Exception as e:
                error_msg = f"Unexpected error in step {step_index + 1}: {str(e)}"
                logging.error(error_msg)
                import traceback
                logging.error(traceback.format_exc())
                
                step_results.append(StepResult(
                    step_index=step_index,
                    task=step_dict.get("task", ""),
                    player_role=step_dict.get("player", ""),
                    individual_results=[],
                    debate_rounds_completed=0,
                    consolidated_result="",
                    artifacts={},
                    success=False,
                    error=error_msg
                ))
        
        # Determine overall success
        successful_steps = sum(1 for r in step_results if r.success)
        overall_success = successful_steps == len(plan_steps)
        
        # Get relationships from datasource
        relationships = [r.to_dict() for r in datasource.get_relationships()]
        
        logging.info("")
        logging.info("=" * 60)
        logging.info("PLAN EXECUTION COMPLETE")
        logging.info(f"Steps completed: {successful_steps}/{len(plan_steps)}")
        logging.info(f"Overall success: {overall_success}")
        if datasource.is_multi_table:
            logging.info(f"Tables: {datasource.tables}")
            logging.info(f"Relationships: {len(relationships)}")
        logging.info("=" * 60)
        
        return ExecutionResult(
            plan_steps_count=len(plan_steps),
            steps_completed=successful_steps,
            step_results=step_results,
            final_workspace=self._filter_workspace(workspace),
            final_metadata=self._extract_final_metadata(workspace, datasource),
            datasource_info=datasource.to_dict(),
            table_metadata=table_metadata,
            relationships=relationships,
            success=overall_success,
            error=None if overall_success else "Some steps failed"
        )
    
    def _filter_workspace(self, workspace: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out internal workspace keys."""
        return {
            k: v for k, v in workspace.items() 
            if not k.startswith("_")
        }
    
    def _extract_final_metadata(
        self, 
        workspace: Dict[str, Any],
        datasource: DataSource
    ) -> Dict[str, Any]:
        """
        Extract the final metadata from the workspace.
        
        This combines all artifacts into a structured metadata output.
        
        Args:
            workspace: The final workspace with all artifacts
            datasource: The DataSource that was analyzed
            
        Returns:
            Structured metadata dictionary
        """
        filtered = self._filter_workspace(workspace)
        
        if datasource.is_multi_table:
            # Organize artifacts by table
            table_artifacts = {}
            dataset_artifacts = {}
            
            for key, value in filtered.items():
                if ":" in key:
                    # Table-namespaced artifact
                    table, artifact = key.split(":", 1)
                    if table not in table_artifacts:
                        table_artifacts[table] = {}
                    table_artifacts[table][artifact] = value
                else:
                    # Dataset-level artifact
                    dataset_artifacts[key] = value
            
            return {
                "type": "multi_table",
                "name": datasource.name,
                "source_type": datasource.source_type.value,
                "tables": table_artifacts,
                "dataset_level": dataset_artifacts,
                "artifact_count": len(filtered)
            }
        else:
            return {
                "type": "single_table",
                "name": datasource.name,
                "source_type": datasource.source_type.value,
                "table": datasource.tables[0] if datasource.tables else None,
                "artifacts": filtered,
                "artifact_count": len(filtered)
            }


def execute_plan(
    plan: Plan,
    datasource: DataSource,
    datasource_key: str,
    metadata_standard: str,
    topology_name: str = "default"
) -> ExecutionResult:
    """
    Convenience function to execute a plan.
    
    Args:
        plan: The Plan object to execute
        datasource: The DataSource to analyze
        datasource_key: Key for the DataSource in the tool registry
        metadata_standard: The metadata standard to follow
        topology_name: Name of the execution topology
        
    Returns:
        ExecutionResult with all results
    """
    executor = PlanExecutor(topology_name=topology_name)
    return executor.execute(
        plan=plan,
        datasource=datasource,
        datasource_key=datasource_key,
        metadata_standard=metadata_standard
    )
