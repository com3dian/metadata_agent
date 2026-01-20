"""
Main Orchestrator - Coordinates planning and execution.

This module provides the Orchestrator class that:
1. Accepts any data source (file, files, SQLite, etc.) via unified DataSource
2. Generates a plan based on the data source and metadata standard
3. Executes the plan using the configured topology
4. Returns the final metadata result

Uses the unified DataSource abstraction for all data access.
This is the main entry point for the multi-agent system.
"""
import logging
import uuid
from typing import Optional, Union, Dict, List, Any

from langchain_core.output_parsers import PydanticOutputParser

from .prompts import get_planning_prompt, get_multi_table_planning_prompt
from .schemas import Plan, ExecutionResult
from .plan_executor import PlanExecutor
from ..datasource import DataSource, DataSourceFactory, create_datasource
from ..players import Player, create_player_from_config, PLAYER_CONFIGS
from ..topology import EXECUTION_TOPOLOGIES
from ..tools.datasource_tools import register_datasource, clear_registry
from ..config import (
    PLANNING_TEMPERATURE, 
    DEFAULT_TOPOLOGY,
    create_llm,
    get_model_name,
    LLM_PROVIDER
)


class Orchestrator:
    """
    The main orchestrator that coordinates plan generation and execution.
    
    The orchestrator:
    1. Takes any data source (auto-detected or explicit DataSource)
    2. Generates a step-by-step plan using available players
    3. Executes the plan with parallel players and debates per step
    4. Returns the extracted metadata
    
    Examples:
        orchestrator = Orchestrator()
        
        # Single file
        result = orchestrator.run("./data/users.csv", standard)
        
        # Multiple files
        result = orchestrator.run(
            {"users": "./users.csv", "orders": "./orders.csv"}, 
            standard
        )
        
        # SQLite database
        result = orchestrator.run("./data/mydb.sqlite", standard)
        
        # Directory
        result = orchestrator.run("./data/my_dataset/", standard)
        
        # Explicit DataSource
        ds = CSVDataSource(...)
        result = orchestrator.run(ds, standard)
    """
    
    def __init__(
        self,
        topology_name: str = None,
        model_name: str = None,
        temperature: float = None,
        provider: str = None
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            topology_name: Name of the execution topology to use (default from config)
            model_name: LLM model for planning (default from config)
            temperature: LLM temperature for planning (default from config)
            provider: LLM provider to use (default from config)
        """
        # Use config defaults if not specified
        topology_name = topology_name or DEFAULT_TOPOLOGY
        temperature = temperature if temperature is not None else PLANNING_TEMPERATURE
        provider = provider or LLM_PROVIDER
        
        if topology_name not in EXECUTION_TOPOLOGIES:
            available = list(EXECUTION_TOPOLOGIES.keys())
            raise ValueError(
                f"Unknown topology '{topology_name}'. Available: {available}"
            )
        
        self.topology_name = topology_name
        self.topology = EXECUTION_TOPOLOGIES[topology_name]
        self.provider = provider
        
        # Planning components - use factory to create LLM
        self.llm = create_llm(
            model_name=model_name,
            temperature=temperature,
            provider=provider
        )
        self.parser = PydanticOutputParser(pydantic_object=Plan)
        self.prompt_template = get_planning_prompt()
        self.planning_chain = self.prompt_template | self.llm | self.parser
        
        # Executor for running the plan
        self.executor = PlanExecutor(topology_name=topology_name)
        
        logging.info(f"Orchestrator initialized with topology: {topology_name}")
    
    def _get_effective_player_pool(self, datasource: DataSource = None) -> list:
        """
        Get the effective player pool, auto-adding relationship_analyst for multi-table.
        
        Args:
            datasource: Optional DataSource to check if multi-table
            
        Returns:
            List of player role names
        """
        player_pool = list(self.topology.get("player_pool", []))
        
        # Auto-add relationship_analyst for multi-table datasets
        if datasource and datasource.is_multi_table:
            if "relationship_analyst" not in player_pool:
                player_pool.append("relationship_analyst")
                logging.info("Auto-added 'relationship_analyst' for multi-table dataset")
        
        return player_pool
    
    def _generate_player_manifest(self, datasource: DataSource = None) -> str:
        """
        Generate a manifest of available players for the planner.
        
        Args:
            datasource: Optional DataSource to determine if relationship tools needed
        
        Returns:
            String description of available players and their capabilities
        """
        player_pool = self._get_effective_player_pool(datasource)
        
        manifest_parts = []
        for role_name in player_pool:
            if role_name in PLAYER_CONFIGS:
                config = PLAYER_CONFIGS[role_name]
                player = create_player_from_config(config, name=role_name)
                manifest_parts.append(player.get_tool_manifest())
        
        return "\n\n".join(manifest_parts)
    
    def _generate_datasource_info(self, datasource: DataSource) -> str:
        """
        Generate a description of the DataSource for the planner.
        
        Args:
            datasource: The DataSource to describe
            
        Returns:
            String description of the data source
        """
        info_parts = [
            f"Dataset Name: {datasource.name}",
            f"Source Type: {datasource.source_type.value}",
            f"Multi-table: {datasource.is_multi_table}",
            f"Tables: {', '.join(datasource.tables)}"
        ]
        
        if datasource.description:
            info_parts.insert(1, f"Description: {datasource.description}")
        
        info_parts.append("\nTable Details:")
        for table in datasource.tables:
            try:
                table_info = datasource.get_table_info(table)
                info_parts.append(
                    f"  - {table}: {table_info.row_count} rows, "
                    f"{table_info.column_count} columns ({', '.join(table_info.column_names[:5])}{'...' if len(table_info.column_names) > 5 else ''})"
                )
            except Exception:
                info_parts.append(f"  - {table}: (info unavailable)")
        
        # Include discovered relationships
        relationships = datasource.get_relationships()
        if relationships:
            info_parts.append("\nDiscovered Relationships:")
            for rel in relationships[:5]:  # Limit to first 5
                info_parts.append(
                    f"  - {rel.from_table}.{rel.from_column} -> "
                    f"{rel.to_table}.{rel.to_column} ({rel.relationship_type})"
                )
            if len(relationships) > 5:
                info_parts.append(f"  ... and {len(relationships) - 5} more")
        
        return "\n".join(info_parts)
    
    def generate_plan(
        self,
        datasource: DataSource,
        metadata_standard: str
    ) -> Optional[Plan]:
        """
        Generate a plan for extracting metadata from a DataSource.
        
        Args:
            datasource: The DataSource to analyze
            metadata_standard: The metadata standard to follow
            
        Returns:
            Plan object with steps, or None if generation failed
        """
        is_multi_table = datasource.is_multi_table
        
        logging.info("=" * 60)
        logging.info("GENERATING PLAN")
        logging.info(f"Dataset: {datasource.name}")
        logging.info(f"Source type: {datasource.source_type.value}")
        logging.info(f"Tables: {datasource.tables}")
        logging.info(f"Multi-table: {is_multi_table}")
        logging.info("=" * 60)
        
        manifest = self._generate_player_manifest(datasource)
        datasource_info = self._generate_datasource_info(datasource)
        
        logging.info("DataSource info:")
        logging.info(datasource_info)
        logging.info("-" * 40)
        logging.info("Available players manifest:")
        logging.info(manifest)
        logging.info("-" * 40)
        
        try:
            if is_multi_table:
                # Use multi-table planning prompt
                multi_table_prompt = get_multi_table_planning_prompt()
                multi_table_chain = multi_table_prompt | self.llm | self.parser
                
                prompt_inputs = {
                    "dataset_info": datasource_info,
                    "dataset_name": datasource.name,
                    "table_names": ", ".join(datasource.tables),
                    "file_type": datasource.source_type.value.upper(),
                    "available_players": manifest,
                    "metadata_standard": metadata_standard,
                }
                
                generated_plan = multi_table_chain.invoke(prompt_inputs)
            else:
                # Use single-table planning prompt
                prompt_inputs = {
                    "file_type": datasource.source_type.value.upper(),
                    "available_players": manifest,
                    "metadata_standard": metadata_standard,
                }
                
                generated_plan = self.planning_chain.invoke(prompt_inputs)
            
            logging.info("Plan generated successfully!")
            logging.info(f"Number of steps: {len(generated_plan.steps)}")
            for i, step in enumerate(generated_plan.steps):
                target_info = f" (tables: {step.target_tables})" if step.target_tables else ""
                logging.info(f"  Step {i+1}: {step.task} (player: {step.player}){target_info}")
            
            return generated_plan
            
        except Exception as e:
            logging.error(f"Plan generation failed: {e}")
            
            # Try to get raw output for debugging
            try:
                if is_multi_table:
                    raw_output = (multi_table_prompt | self.llm).invoke(prompt_inputs)
                else:
                    raw_output = (self.prompt_template | self.llm).invoke(prompt_inputs)
                logging.error(f"Raw LLM output: {raw_output}")
            except Exception:
                pass
            
            return None
    
    def execute_plan(
        self,
        plan: Plan,
        datasource: DataSource,
        metadata_standard: str
    ) -> ExecutionResult:
        """
        Execute a generated plan.
        
        Args:
            plan: The Plan to execute
            datasource: The DataSource to analyze
            metadata_standard: The metadata standard to follow
            
        Returns:
            ExecutionResult with all step results and final metadata
        """
        # Register the datasource for tools to access
        datasource_key = f"ds_{uuid.uuid4().hex[:8]}"
        register_datasource(datasource_key, datasource)
        
        # Get effective player pool (auto-adds relationship_analyst for multi-table)
        effective_player_pool = self._get_effective_player_pool(datasource)
        
        try:
            return self.executor.execute(
                plan=plan,
                datasource=datasource,
                datasource_key=datasource_key,
                metadata_standard=metadata_standard,
                player_pool=effective_player_pool
            )
        finally:
            # Optionally clean up registry after execution
            # clear_registry()  # Uncomment if you want to clean up
            pass
    
    def run(
        self,
        source: Union[str, List[str], Dict[str, str], DataSource],
        metadata_standard: str,
        name: str = "dataset",
        **kwargs
    ) -> Optional[ExecutionResult]:
        """
        Run the complete orchestration: plan generation + execution.
        
        This is the unified entry point for ALL data sources.
        
        Args:
            source: Data source - can be:
                - str: Single file path, directory, or SQLite database
                - List[str]: Multiple file paths
                - Dict[str, str]: Mapping of table names to file paths
                - DataSource: An already-created DataSource instance
            metadata_standard: The metadata standard to follow
            name: Name for the dataset (used if creating new DataSource)
            **kwargs: Additional arguments passed to DataSource creation
            
        Returns:
            ExecutionResult with all results, or None if planning failed
            
        Examples:
            # Single file
            result = orchestrator.run("./data/users.csv", standard)
            
            # Multiple files
            result = orchestrator.run(
                {"users": "./users.csv", "orders": "./orders.csv"},
                standard
            )
            
            # SQLite database
            result = orchestrator.run("./data/mydb.sqlite", standard)
            
            # Directory of CSVs
            result = orchestrator.run("./data/dataset/", standard)
        """
        # Step 0: Create/normalize DataSource
        if isinstance(source, DataSource):
            datasource = source
        else:
            datasource = create_datasource(source, name=name, **kwargs)
        
        logging.info("=" * 60)
        logging.info("STARTING ORCHESTRATION")
        logging.info(f"Dataset: {datasource.name}")
        logging.info(f"Type: {datasource.source_type.value}")
        logging.info(f"Tables: {datasource.tables}")
        logging.info("=" * 60)
        
        # Step 1: Generate the plan
        plan = self.generate_plan(
            datasource=datasource,
            metadata_standard=metadata_standard
        )
        
        if plan is None:
            logging.error("Failed to generate plan. Aborting execution.")
            return None
        
        # Step 2: Execute the plan
        result = self.execute_plan(
            plan=plan,
            datasource=datasource,
            metadata_standard=metadata_standard
        )
        
        return result


# ===================================================================
#  CONVENIENCE FUNCTIONS
# ===================================================================

def run_metadata_extraction(
    source: Union[str, List[str], Dict[str, str], DataSource],
    metadata_standard: str,
    name: str = "dataset",
    topology_name: str = "default",
    **kwargs
) -> Optional[ExecutionResult]:
    """
    Convenience function to run metadata extraction.
    
    The orchestrator automatically adapts for multi-table datasets by
    adding 'relationship_analyst' to the player pool when needed.
    
    Args:
        source: Any supported data source format
        metadata_standard: The metadata standard to follow
        name: Name for the dataset
        topology_name: Execution topology (default: "default")
        **kwargs: Additional arguments for DataSource creation
        
    Returns:
        ExecutionResult or None if failed
        
    Examples:
        # Single file
        result = run_metadata_extraction("./data/users.csv", standard)
        
        # Multiple files (relationship_analyst auto-added)
        result = run_metadata_extraction(
            {"users": "./users.csv", "orders": "./orders.csv"},
            standard
        )
    """
    orchestrator = Orchestrator(topology_name=topology_name)
    return orchestrator.run(
        source=source,
        metadata_standard=metadata_standard,
        name=name,
        **kwargs
    )
