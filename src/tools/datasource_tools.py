"""
Unified DataSource Tools for the Multi-Agent System.

These tools work with the DataSource abstraction, providing a consistent
interface regardless of the underlying data storage (CSV, SQLite, etc.).

All tools receive a DataSource reference (serialized) and operate through
the unified DataSource API.
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

# Global registry for DataSource instances
# Tools receive a reference key and look up the actual DataSource here
_datasource_registry: Dict[str, Any] = {}


def register_datasource(key: str, datasource: Any) -> str:
    """
    Register a DataSource in the global registry.
    Returns the key for later retrieval.
    """
    _datasource_registry[key] = datasource
    return key


def get_datasource(key: str) -> Any:
    """Get a DataSource from the registry."""
    if key not in _datasource_registry:
        raise KeyError(f"DataSource '{key}' not found in registry")
    return _datasource_registry[key]


def clear_registry():
    """Clear all registered DataSources."""
    _datasource_registry.clear()


# ===================================================================
#  DATASET-LEVEL TOOLS
# ===================================================================


@tool
def get_dataset_overview(datasource_key: str) -> Dict[str, Any]:
    """
    Get an overview of the entire dataset including all tables.

    Args:
        datasource_key: Key to the registered DataSource

    Returns:
        Overview with table names, row counts, column info, and relationships
    """
    try:
        ds = get_datasource(datasource_key)

        tables_info = {}
        for table in ds.tables:
            info = ds.get_table_info(table)
            tables_info[table] = {
                "row_count": info.row_count,
                "column_count": info.column_count,
                "columns": info.column_names,
                "primary_key": info.primary_key,
            }

        relationships = [r.to_dict() for r in ds.get_relationships()]

        return {
            "name": ds.name,
            "source_type": ds.source_type.value,
            "is_multi_table": ds.is_multi_table,
            "table_count": len(ds.tables),
            "tables": tables_info,
            "relationships": relationships,
            "relationship_count": len(relationships),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def list_tables(datasource_key: str) -> List[str]:
    """
    List all tables in the dataset.

    Args:
        datasource_key: Key to the registered DataSource

    Returns:
        List of table names
    """
    try:
        ds = get_datasource(datasource_key)
        return ds.tables
    except Exception as e:
        return [f"Error: {str(e)}"]


@tool
def get_dataset_schema(datasource_key: str) -> Dict[str, Any]:
    """
    Get the complete schema of the dataset including tables, columns, and relationships.

    Args:
        datasource_key: Key to the registered DataSource

    Returns:
        Full schema dictionary
    """
    try:
        ds = get_datasource(datasource_key)
        return ds.get_schema()
    except Exception as e:
        return {"error": str(e)}


# ===================================================================
#  TABLE-LEVEL TOOLS
# ===================================================================


@tool
def get_table_info(datasource_key: str, table: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific table.

    Args:
        datasource_key: Key to the registered DataSource
        table: Name of the table

    Returns:
        Table info including row count, columns, types, etc.
    """
    try:
        ds = get_datasource(datasource_key)
        info = ds.get_table_info(table)
        return info.to_dict()
    except Exception as e:
        return {"error": str(e)}


@tool
def get_row_count(datasource_key: str, table: str = None) -> int:
    """
    Get the number of rows in a table.

    Args:
        datasource_key: Key to the registered DataSource
        table: Table name (optional if single-table dataset)

    Returns:
        Row count
    """
    try:
        ds = get_datasource(datasource_key)
        table = table or ds.tables[0]
        info = ds.get_table_info(table)
        return info.row_count
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_column_names(datasource_key: str, table: str = None) -> List[str]:
    """
    Get the column names for a table.

    Args:
        datasource_key: Key to the registered DataSource
        table: Table name (optional if single-table dataset)

    Returns:
        List of column names
    """
    try:
        ds = get_datasource(datasource_key)
        table = table or ds.tables[0]
        info = ds.get_table_info(table)
        return info.column_names
    except Exception as e:
        return [f"Error: {str(e)}"]


@tool
def get_column_types(datasource_key: str, table: str = None) -> Dict[str, str]:
    """
    Get data types for all columns in a table.

    Args:
        datasource_key: Key to the registered DataSource
        table: Table name (optional if single-table dataset)

    Returns:
        Dict mapping column names to their data types
    """
    try:
        ds = get_datasource(datasource_key)
        table = table or ds.tables[0]
        info = ds.get_table_info(table)
        return {col.name: col.dtype for col in info.columns}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_sample_rows(datasource_key: str, table: str = None, n: int = 5) -> str:
    """
    Get a sample of rows from a table.

    Args:
        datasource_key: Key to the registered DataSource
        table: Table name (optional if single-table dataset)
        n: Number of rows to return

    Returns:
        String representation of the sample rows
    """
    try:
        ds = get_datasource(datasource_key)
        table = table or ds.tables[0]
        df = ds.read_table(table, nrows=n)
        return df.to_string()
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_column_statistics(datasource_key: str, table: str = None) -> Dict[str, Any]:
    """
    Get statistics for all columns in a table.

    Args:
        datasource_key: Key to the registered DataSource
        table: Table name (optional if single-table dataset)

    Returns:
        Statistics for each column
    """
    try:
        ds = get_datasource(datasource_key)
        table = table or ds.tables[0]
        df = ds.read_table(table)
        return df.describe(include="all").to_dict()
    except Exception as e:
        return {"error": str(e)}


@tool
def get_missing_values(datasource_key: str, table: str = None) -> Dict[str, int]:
    """
    Get count of missing values per column.

    Args:
        datasource_key: Key to the registered DataSource
        table: Table name (optional if single-table dataset)

    Returns:
        Dict mapping column names to missing value counts
    """
    try:
        ds = get_datasource(datasource_key)
        table = table or ds.tables[0]
        df = ds.read_table(table)
        return df.isnull().sum().to_dict()
    except Exception as e:
        return {"error": str(e)}


@tool
def get_unique_values(
    datasource_key: str, table: str, column: str, limit: int = 100
) -> List[Any]:
    """
    Get unique values from a specific column.

    Args:
        datasource_key: Key to the registered DataSource
        table: Table name
        column: Column name
        limit: Maximum number of unique values to return

    Returns:
        List of unique values
    """
    try:
        ds = get_datasource(datasource_key)
        values = ds.get_column_values(table, column, limit=limit)
        return values
    except Exception as e:
        return [f"Error: {str(e)}"]


# ===================================================================
#  RELATIONSHIP TOOLS
# ===================================================================


@tool
def get_relationships(datasource_key: str) -> List[Dict[str, Any]]:
    """
    Get all discovered or defined relationships between tables.

    Args:
        datasource_key: Key to the registered DataSource

    Returns:
        List of relationships with from/to tables and columns
    """
    try:
        ds = get_datasource(datasource_key)
        return [r.to_dict() for r in ds.get_relationships()]
    except Exception as e:
        return [{"error": str(e)}]


@tool
def analyze_potential_relationship(
    datasource_key: str,
    from_table: str,
    from_column: str,
    to_table: str,
    to_column: str,
) -> Dict[str, Any]:
    """
    Analyze a potential relationship between two columns.

    Args:
        datasource_key: Key to the registered DataSource
        from_table: Source table name
        from_column: Source column name
        to_table: Target table name
        to_column: Target column name

    Returns:
        Analysis including match rate, cardinality, and relationship type
    """
    try:
        ds = get_datasource(datasource_key)

        # Get unique values from both columns
        values_from = set(ds.get_column_values(from_table, from_column))
        values_to = set(ds.get_column_values(to_table, to_column))

        if not values_from or not values_to:
            return {"error": "One or both columns are empty"}

        intersection = values_from & values_to

        # Calculate metrics
        match_rate_from = len(intersection) / len(values_from) if values_from else 0
        match_rate_to = len(intersection) / len(values_to) if values_to else 0

        # Get row counts for cardinality analysis
        info_from = ds.get_table_info(from_table)
        info_to = ds.get_table_info(to_table)

        unique_ratio_from = (
            len(values_from) / info_from.row_count if info_from.row_count else 0
        )
        unique_ratio_to = len(values_to) / info_to.row_count if info_to.row_count else 0

        # Determine relationship type
        if unique_ratio_from > 0.9 and unique_ratio_to > 0.9:
            rel_type = "one-to-one"
        elif unique_ratio_from > 0.9:
            rel_type = "one-to-many"
        elif unique_ratio_to > 0.9:
            rel_type = "many-to-one"
        else:
            rel_type = "many-to-many"

        confidence = (match_rate_from + match_rate_to) / 2

        return {
            "from_table": from_table,
            "from_column": from_column,
            "to_table": to_table,
            "to_column": to_column,
            "unique_values_from": len(values_from),
            "unique_values_to": len(values_to),
            "common_values": len(intersection),
            "match_rate_from": round(match_rate_from, 4),
            "match_rate_to": round(match_rate_to, 4),
            "is_unique_in_from": unique_ratio_from > 0.9,
            "is_unique_in_to": unique_ratio_to > 0.9,
            "suggested_type": rel_type,
            "confidence": round(confidence, 4),
            "likely_valid": confidence > 0.5
            and (unique_ratio_from > 0.9 or unique_ratio_to > 0.9),
            "sample_common_values": list(intersection)[:10],
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def preview_join(
    datasource_key: str,
    from_table: str,
    from_column: str,
    to_table: str,
    to_column: str,
    join_type: str = "inner",
    n_rows: int = 5,
) -> str:
    """
    Preview the result of joining two tables.

    Args:
        datasource_key: Key to the registered DataSource
        from_table: First table
        from_column: Join column in first table
        to_table: Second table
        to_column: Join column in second table
        join_type: Type of join (inner, left, right, outer)
        n_rows: Number of preview rows

    Returns:
        String representation of joined preview
    """
    try:
        import pandas as pd

        ds = get_datasource(datasource_key)

        df_from = ds.read_table(from_table, nrows=1000)
        df_to = ds.read_table(to_table, nrows=1000)

        joined = pd.merge(
            df_from,
            df_to,
            left_on=from_column,
            right_on=to_column,
            how=join_type,
            suffixes=(f"_{from_table}", f"_{to_table}"),
        )

        result = f"Join: {from_table}.{from_column} {join_type.upper()} {to_table}.{to_column}\n"
        result += (
            f"Result: {len(joined)} rows from {len(df_from)} x {len(df_to)} samples\n"
        )
        result += f"Columns: {list(joined.columns)}\n\n"
        result += joined.head(n_rows).to_string()

        return result
    except Exception as e:
        return f"Error: {str(e)}"


# ===================================================================
#  CROSS-TABLE ANALYSIS TOOLS
# ===================================================================


@tool
def find_common_columns(datasource_key: str) -> Dict[str, Any]:
    """
    Find columns that appear in multiple tables (potential join keys).

    Args:
        datasource_key: Key to the registered DataSource

    Returns:
        Columns grouped by how many tables they appear in
    """
    try:
        ds = get_datasource(datasource_key)

        column_to_tables: Dict[str, List[str]] = {}

        for table in ds.tables:
            info = ds.get_table_info(table)
            for col in info.column_names:
                col_lower = col.lower().strip()
                if col_lower not in column_to_tables:
                    column_to_tables[col_lower] = []
                column_to_tables[col_lower].append(f"{table}.{col}")

        shared = {
            col: tables for col, tables in column_to_tables.items() if len(tables) > 1
        }

        return {
            "shared_columns": shared,
            "shared_count": len(shared),
            "potential_join_keys": list(shared.keys()),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def compare_table_schemas(datasource_key: str) -> Dict[str, Any]:
    """
    Compare schemas across all tables to identify patterns.

    Args:
        datasource_key: Key to the registered DataSource

    Returns:
        Schema comparison including common columns, unique columns, etc.
    """
    try:
        ds = get_datasource(datasource_key)

        schemas = {}
        all_columns = set()

        for table in ds.tables:
            info = ds.get_table_info(table)
            schemas[table] = {
                "columns": {
                    col.name: {
                        "dtype": col.dtype,
                        "nullable": col.nullable,
                        "is_primary_key": col.is_primary_key,
                        "is_foreign_key": col.is_foreign_key,
                    }
                    for col in info.columns
                },
                "column_count": info.column_count,
                "row_count": info.row_count,
            }
            all_columns.update(info.column_names)

        # Find ID-like columns
        id_columns = [
            c
            for c in all_columns
            if c.lower().endswith("id") or c.lower().endswith("_id")
        ]

        # Find date-like columns
        date_keywords = ["date", "time", "created", "updated", "timestamp"]
        date_columns = [
            c for c in all_columns if any(kw in c.lower() for kw in date_keywords)
        ]

        return {
            "schemas": schemas,
            "total_unique_columns": len(all_columns),
            "potential_id_columns": id_columns,
            "potential_date_columns": date_columns,
            "all_columns": sorted(list(all_columns)),
        }
    except Exception as e:
        return {"error": str(e)}


# ===================================================================
#  CONVENIENCE: GET ALL TOOLS
# ===================================================================


def get_all_datasource_tools() -> List:
    """Return list of all DataSource tools."""
    return [
        # Dataset-level
        get_dataset_overview,
        list_tables,
        get_dataset_schema,
        # Table-level
        get_table_info,
        get_row_count,
        get_column_names,
        get_column_types,
        get_sample_rows,
        get_column_statistics,
        get_missing_values,
        get_unique_values,
        # Relationships
        get_relationships,
        analyze_potential_relationship,
        preview_join,
        # Cross-table
        find_common_columns,
        compare_table_schemas,
    ]


def get_single_table_tools() -> List:
    """Return tools appropriate for single-table datasets."""
    return [
        get_dataset_overview,
        get_table_info,
        get_row_count,
        get_column_names,
        get_column_types,
        get_sample_rows,
        get_column_statistics,
        get_missing_values,
        get_unique_values,
    ]


def get_multi_table_tools() -> List:
    """Return tools appropriate for multi-table datasets."""
    return get_all_datasource_tools()
