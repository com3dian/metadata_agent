"""
This script demonstrates the new orchestrator architecture.

It shows both:
1. Plan generation only (for inspection)
2. Full execution (plan + parallel players + debate)

Usage:
    python run_planner.py
"""
import sys
import os
import logging
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.orchestrator.orchestrator import Orchestrator
from src.standards import METADATA_STANDARDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def demo_plan_generation():
    """
    Demonstrate plan generation without execution.
    """
    print("=" * 60)
    print("DEMO: PLAN GENERATION ONLY")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = Orchestrator(topology_name="default")
    
    # Generate a plan
    plan = orchestrator.generate_plan(
        file_type="CSV",
        metadata_standard=METADATA_STANDARDS["basic"]
    )
    
    if plan:
        print("\n--- Generated Plan ---")
        for i, step in enumerate(plan.steps):
            print(f"\nStep {i + 1}:")
            print(f"  Task: {step.task}")
            print(f"  Player: {step.player}")
            print(f"  Rationale: {step.rationale}")
            print(f"  Inputs: {step.inputs}")
            print(f"  Outputs: {step.outputs}")
    else:
        print("Failed to generate plan.")
    
    return plan


def demo_full_execution():
    """
    Demonstrate full execution with planning + parallel players + debate.
    """
    print("\n" + "=" * 60)
    print("DEMO: FULL EXECUTION")
    print("=" * 60)
    
    # Configuration
    DATASET_PATH = "./data/test_data.csv"
    FILE_TYPE = "CSV"
    METADATA_STANDARD = "basic"
    TOPOLOGY = "fast"  # Use 'fast' for quicker demo, 'default' for full
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  File Type: {FILE_TYPE}")
    print(f"  Metadata Standard: {METADATA_STANDARD}")
    print(f"  Topology: {TOPOLOGY}")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\nWarning: Dataset not found at {DATASET_PATH}")
        print("Creating a sample dataset for demo...")
        os.makedirs("./data", exist_ok=True)
        import pandas as pd
        sample_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "city": ["NYC", "LA", "Chicago", "NYC", "Boston"]
        })
        sample_df.to_csv(DATASET_PATH, index=False)
        print(f"Sample dataset created at {DATASET_PATH}")
    
    # Initialize and run
    orchestrator = Orchestrator(topology_name=TOPOLOGY)
    
    result = orchestrator.run(
        file_type=FILE_TYPE,
        dataset_path=DATASET_PATH,
        metadata_standard=METADATA_STANDARDS[METADATA_STANDARD]
    )
    
    if result:
        print("\n" + "=" * 60)
        print("EXECUTION RESULTS")
        print("=" * 60)
        print(f"\nSuccess: {result.success}")
        print(f"Steps Completed: {result.steps_completed}/{result.plan_steps_count}")
        
        print("\n--- Step Results Summary ---")
        for step_result in result.step_results:
            print(f"\nStep {step_result.step_index + 1}: {step_result.task}")
            print(f"  Player Role: {step_result.player_role}")
            print(f"  Success: {step_result.success}")
            print(f"  Debate Rounds: {step_result.debate_rounds_completed}")
            print(f"  Artifacts: {list(step_result.artifacts.keys())}")
        
        print("\n--- Final Metadata ---")
        pprint(result.final_metadata)
    else:
        print("Execution failed.")
    
    return result


def main():
    """
    Main demo function.
    """
    print("=" * 60)
    print("METADATA AGENT - ORCHESTRATOR DEMO")
    print("=" * 60)
    
    # Run demos
    # demo_plan_generation()
    demo_full_execution()


if __name__ == "__main__":
    main()
