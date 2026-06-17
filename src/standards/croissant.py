"""
Necessary definitions for the Croissant metadata standard.

Subset of the Croissant standard that is findable across common datasets.
Full Croissant is kept for future use.
"""

from typing import Any, Dict, List, Optional

from src.standards.schema_builder import build_schema_for_standard


file_set_standard: Dict[str, Dict[str, Any]] = {
    "includes": {
        "type": List[str],
        "default": [],
        "description": "Glob patterns or file names specifying files to include.",
        "prompt_hint": "Examples: ['*.csv', 'data/**/*.csv'] or ['biota.csv', 'samples.csv'].",
    },
    "excludes": {
        "type": List[str],
        "default": [],
        "description": "Glob patterns or file names specifying files to exclude.",
        "prompt_hint": "Examples: ['temp/*', '**/*.bak'] or ['README.md'].",
    },
}

record_set_standard: Dict[str, Dict[str, Any]] = {
    "field": {
        "type": "CroissantField",
        "default": ...,
        "description": "Field metadata describing a record.",
        "prompt_hint": (
            "Field object with source, dataType, isArray, arrayShape, references."
        ),
    },
    "key": {
        "type": Optional[str],
        "default": None,
        "description": "Primary key field for records.",
        "prompt_hint": "Field uniquely identifying a record.",
    },
    "data": {
        "type": List[str],
        "default": [],
        "description": "References to fields that compose the record.",
        "prompt_hint": "List of field identifiers.",
    },
    "examples": {
        "type": List[str],
        "default": [],
        "description": "Example record values.",
        "prompt_hint": "Representative examples of records.",
    },
    "annotation": {
        "type": Optional[str],
        "default": None,
        "description": "Additional annotation or comments.",
        "prompt_hint": "Optional notes about the RecordSet.",
    },
}

field_standard: Dict[str, Dict[str, Any]] = {
    "source": {
        "type": str,
        "default": ...,
        "description": "Source column or field path.",
        "prompt_hint": "Column name in the source CSV.",
    },
    "dataType": {
        "type": str,
        "default": ...,
        "description": "Data type of the field.",
        "prompt_hint": "string, integer, float, boolean, date, datetime, etc.",
    },
    "isArray": {
        "type": bool,
        "default": False,
        "description": (
            "Whether the field is an array of values. "
            "If true and arrayShape is not specified, the default shape is (-1,)."
        ),
        "prompt_hint": "Set true for list-valued fields.",
    },
    "arrayShape": {
        "type": Optional[str],
        "default": None,
        "description": (
            "Shape of the array as a comma-separated string. "
            "-1 indicates an unknown dimension size. "
            "Example: '(-1,)', '(3,224,224)'."
        ),
        "prompt_hint": "Specify only when isArray is true.",
    },
    "references": {
        "type": Optional[str],
        "default": None,
        "description": (
            "Reference to another field in another RecordSet. "
            "Equivalent to a foreign key relationship."
        ),
        "prompt_hint": "Format: 'users.user_id' or 'RecordSet.field'.",
    },
}


# date modified ?
# spatial temporal resolution

# 

croissant_standard_subset: Dict[str, Dict[str, Any]] = {
    "name": {
        "type": str,
        "default": ...,
        "description": "Dataset name",
        "prompt_hint": "Name of the dataset.",
    },
    "description": {
        "type": Optional[str],
        "default": None,
        "description": "Dataset description",
        "prompt_hint": "Explain what the dataset contains.",
    },
    "keywords": {
        "type": List[str],
        "default": [],
        "description": "Keywords associated with the dataset",
        "prompt_hint": "List search terms or tags.",
    },
    "filesets": {
        "type": List["FileSet"],
        "default": [],
        "description": "Physical files belonging to the dataset",
        "prompt_hint": "One object per file or file collection.",
    },
    "recordsets": {
        "type": List["RecordSet"],
        "default": [],
        "description": "Logical tables contained in the dataset",
        "prompt_hint": "One object per table or entity.",
    },
    "inLanguage": {
        "type": List[str],
        "default": [],
        "description": "Language(s) used in the dataset content.",
        "prompt_hint": (
            "Specify ISO 639 language codes appearing in the data, "
            "e.g. ['en'], ['en', 'fr'], ['zh-CN']."
        ),
    },
    "spatialCoverage": {
        "type": Optional[Dict[str, float]],
        "default": None,
        "description": (
            "Geographic bounding box with keys: "
            "min_lat, min_lon, max_lat, max_lon"
        ),
        "prompt_hint": (
            "Geographic bounding box in WGS84 with numeric fields: "
            "min_lat, min_lon, max_lat, max_lon"
        ),
    },
    "temporalCoverage": {
        "type": Optional[str],
        "default": None,
        "description": "Time period covered, from and to date",
        "prompt_hint": "Time period covered, from and to date",
    },
}


FileSet = build_schema_for_standard(
    "file_set",
    file_set_standard,
    model_name="FileSet",
)
CroissantField = build_schema_for_standard(
    "croissant_field",
    field_standard,
    model_name="CroissantField",
)
RecordSet = build_schema_for_standard(
    "record_set",
    record_set_standard,
    model_name="RecordSet",
    model_registry={"CroissantField": CroissantField},
)

CROISSANT_MODEL_REGISTRY = {
    "FileSet": FileSet,
    "RecordSet": RecordSet,
}

CroissantStandardSubsetMetadata = build_schema_for_standard(
    "croissant_standard_subset",
    croissant_standard_subset,
    model_name="CroissantStandardSubsetMetadata",
    model_registry=CROISSANT_MODEL_REGISTRY,
)
