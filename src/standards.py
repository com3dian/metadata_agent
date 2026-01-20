"""
This file contains predefined, named metadata standards that can be used by the orchestrator.

Includes standards for both single-file and multi-file (relational) datasets.
"""

METADATA_STANDARDS = {
    "basic": """
{
    "title": "...",
    "description": "...",
    "schema": {
        "fields": [
            {
                "name": "...",
                "type": "...",
                "description": "..."
            }
        ]
    }
}
""",
    "dublin_core": """
{
    "title": "...",
    "creator": "...",
    "subject": "...",
    "description": "...",
    "publisher": "...",
    "date": "...",
    "type": "...",
    "format": "...",
    "identifier": "...",
    "language": "..."
}
""",
    # Standard for multi-file/relational datasets
    "relational": """
{
    "dataset": {
        "name": "...",
        "description": "...",
        "domain": "...",
        "created_date": "...",
        "version": "..."
    },
    "tables": [
        {
            "name": "...",
            "description": "...",
            "row_count": 0,
            "primary_key": "...",
            "fields": [
                {
                    "name": "...",
                    "type": "...",
                    "description": "...",
                    "nullable": true,
                    "is_primary_key": false,
                    "is_foreign_key": false
                }
            ]
        }
    ],
    "relationships": [
        {
            "name": "...",
            "description": "...",
            "from_table": "...",
            "from_column": "...",
            "to_table": "...",
            "to_column": "...",
            "relationship_type": "one-to-many | many-to-one | one-to-one | many-to-many",
            "cardinality": "...",
            "is_mandatory": true
        }
    ],
    "data_quality": {
        "completeness": "...",
        "consistency": "...",
        "notes": "..."
    }
}
""",
    # Simplified relational standard for quick analysis
    "relational_simple": """
{
    "dataset_name": "...",
    "description": "...",
    "tables": {
        "table_name": {
            "description": "...",
            "columns": ["col1", "col2", ...],
            "row_count": 0,
            "key_columns": ["..."]
        }
    },
    "relationships": [
        {
            "from": "table.column",
            "to": "table.column",
            "type": "one-to-many"
        }
    ]
}
""",
    # Ecological/scientific data standard (relevant for your biota/observation data)
    "ecological_data": """
{
    "dataset": {
        "title": "...",
        "description": "...",
        "geographic_coverage": {
            "location": "...",
            "bounding_box": {
                "north": 0.0,
                "south": 0.0,
                "east": 0.0,
                "west": 0.0
            }
        },
        "temporal_coverage": {
            "start_date": "...",
            "end_date": "..."
        },
        "taxonomic_coverage": ["..."],
        "methods": "..."
    },
    "tables": [
        {
            "name": "...",
            "description": "...",
            "entity_type": "observation | sample | measurement | taxon | location",
            "row_count": 0,
            "fields": [
                {
                    "name": "...",
                    "type": "...",
                    "description": "...",
                    "unit": "...",
                    "standard_term": "..."
                }
            ]
        }
    ],
    "relationships": [
        {
            "description": "...",
            "from_table": "...",
            "from_column": "...",
            "to_table": "...",
            "to_column": "...",
            "relationship_type": "..."
        }
    ],
    "data_standards": {
        "taxonomy": "GBIF | WoRMS | ITIS | ...",
        "coordinates": "WGS84 | ...",
        "units": "SI | ..."
    }
}
"""
}
