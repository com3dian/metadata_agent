"""
Tools module for the multi-agent system.

This module provides tools that players can use to analyze datasets.

## DataSource Tools (Recommended)

These tools work with the DataSource abstraction, providing a consistent
interface regardless of the underlying data storage (CSV, SQLite, etc.):

Dataset-level:
- get_dataset_overview: Overview of the entire dataset
- list_tables: List all tables
- get_dataset_schema: Complete schema with relationships

Table-level:
- get_table_info: Detailed table information
- get_row_count: Row count for a table
- get_column_names: Column names
- get_column_types: Column data types
- get_sample_rows: Preview rows
- get_column_statistics: Statistics for all columns
- get_missing_values: Missing value counts
- get_unique_values: Unique values in a column

Relationship tools:
- get_relationships: Get discovered/defined relationships
- analyze_potential_relationship: Analyze a specific relationship
- preview_join: Preview joining two tables
- find_common_columns: Find shared columns across tables
- compare_table_schemas: Compare table structures
"""

from . import datasource_tools

# Note: pandas_tools is deprecated - import only if needed for backwards compat
# from . import pandas_tools  # DEPRECATED: use datasource_tools instead

from .datasource_tools import (
    # Registry functions
    register_datasource,
    get_datasource,
    clear_registry,
    # Tool collections
    get_all_datasource_tools,
    get_single_table_tools,
    get_multi_table_tools,
    # Individual tools
    get_dataset_overview,
    list_tables,
    get_dataset_schema,
    get_table_info,
    get_row_count,
    get_column_names,
    get_column_types,
    get_sample_rows,
    get_column_statistics,
    get_missing_values,
    get_unique_values,
    get_relationships,
    analyze_potential_relationship,
    preview_join,
    find_common_columns,
    compare_table_schemas,
)

__all__ = [
    # Module
    "datasource_tools",
    # Registry
    "register_datasource",
    "get_datasource", 
    "clear_registry",
    # Tool collections
    "get_all_datasource_tools",
    "get_single_table_tools",
    "get_multi_table_tools",
    # Individual tools
    "get_dataset_overview",
    "list_tables",
    "get_dataset_schema",
    "get_table_info",
    "get_row_count",
    "get_column_names",
    "get_column_types",
    "get_sample_rows",
    "get_column_statistics",
    "get_missing_values",
    "get_unique_values",
    "get_relationships",
    "analyze_potential_relationship",
    "preview_join",
    "find_common_columns",
    "compare_table_schemas",
]
