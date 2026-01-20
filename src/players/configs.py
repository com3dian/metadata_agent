"""
Player configurations for the multi-agent system.

This module defines the available player roles with their prompts and tools.
Players are instantiated from these configs at runtime.

Uses the unified DataSource tools for all data access.

Note: model_name and temperature are optional - if not specified,
the defaults from config.py will be used.
"""
from typing import Dict, Any

from ..tools import datasource_tools


PLAYER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "data_analyst": {
        "role_prompt": (
            "You are an expert data analyst. Your job is to perform statistical "
            "analysis on datasets, identify patterns, and extract meaningful insights. "
            "Focus on numerical summaries, distributions, and data quality. "
            "For multi-table datasets, analyze each table's characteristics and note "
            "potential relationships between tables."
        ),
        "tools": [
            datasource_tools.get_dataset_overview,
            datasource_tools.get_table_info,
            datasource_tools.get_row_count, 
            datasource_tools.get_column_names,
            datasource_tools.get_column_statistics,
            datasource_tools.get_missing_values,
            datasource_tools.find_common_columns,
        ],
        # model_name: uses config default
        # temperature: uses config default
    },
    "schema_expert": {
        "role_prompt": (
            "You are a database schema expert. Your job is to describe the structure "
            "of datasets, including column names, data types, relationships between "
            "fields, and recommend appropriate metadata schemas. For multi-table "
            "datasets, identify primary keys, foreign keys, and normalization patterns."
        ),
        "tools": [
            datasource_tools.get_dataset_schema,
            datasource_tools.get_column_names,
            datasource_tools.get_column_types,
            datasource_tools.get_sample_rows,
            datasource_tools.compare_table_schemas,
            datasource_tools.find_common_columns,
        ],
    },
    "metadata_specialist": {
        "role_prompt": (
            "You are a metadata specialist familiar with standards like Dublin Core, "
            "DCAT, and Schema.org. Your job is to extract metadata as STRUCTURED "
            "field-value pairs. Output only the metadata fields and their values in "
            "a clean, compact format. Avoid lengthy explanations - focus on populating "
            "metadata fields according to the specified standard. For multi-table "
            "datasets, include relationship metadata and per-table descriptions."
        ),
        "tools": [
            datasource_tools.get_dataset_overview,
            datasource_tools.get_dataset_schema,
        ],
        "temperature": 0.3,  # Lower for more consistent, structured output
    },
    "critic": {
        "role_prompt": (
            "You are a meticulous quality assurance critic. Your job is to review "
            "analyses from other agents, identify flaws, omissions, inconsistencies, "
            "and suggest improvements. You focus on accuracy and completeness. "
            "For multi-table analysis, verify that relationships are correctly "
            "identified and that cross-table consistency is maintained."
        ),
        "tools": [],
        "temperature": 0.4,
    },
    # Specialized player for relationship analysis
    "relationship_analyst": {
        "role_prompt": (
            "You are a database relationship expert specializing in discovering and "
            "validating relationships between tables in multi-table datasets. Your job "
            "is to identify primary keys, foreign keys, and the nature of relationships "
            "(one-to-one, one-to-many, many-to-many). You analyze column name patterns, "
            "data type compatibility, and value overlaps to determine how tables connect. "
            "Output relationships in a structured format suitable for metadata records."
        ),
        "tools": [
            datasource_tools.get_relationships,
            datasource_tools.find_common_columns,
            datasource_tools.analyze_potential_relationship,
            datasource_tools.preview_join,
            datasource_tools.compare_table_schemas,
            datasource_tools.get_dataset_overview,
        ],
        "temperature": 0.3,
    },
}
