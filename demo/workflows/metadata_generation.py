import hashlib
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.orchestrator import run_metadata_extraction
from src.standards import METADATA_STANDARDS


SUPPORTED_FILE_TYPES = ["csv", "tsv", "txt"]


def available_metadata_standards() -> list[str]:
    return list(METADATA_STANDARDS)


def uploaded_file_key(file_bytes: bytes, standard_name: str) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"{digest}:{standard_name}"


def load_preview(file_name: str, file_bytes: bytes, rows: int = 25) -> pd.DataFrame:
    separator = "\t" if file_name.lower().endswith(".tsv") else ","
    return pd.read_csv(BytesIO(file_bytes), sep=separator, nrows=rows)


def generate_metadata(file_name: str, file_bytes: bytes, standard_name: str) -> dict[str, Any]:
    if standard_name not in METADATA_STANDARDS:
        raise ValueError(f"Unknown metadata standard: {standard_name}")

    temp_path = _write_upload_to_temp(file_name, file_bytes)
    try:
        result = run_metadata_extraction(
            source=temp_path,
            metadata_standard=METADATA_STANDARDS[standard_name],
            metadata_standard_name=standard_name,
            name=Path(file_name).stem,
        )
        if result is None:
            raise RuntimeError("Metadata agent did not return a result.")
        return _to_displayable(result)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def extract_metadata(result: dict[str, Any]) -> Any:
    return result.get("final_metadata") or result.get("final_workspace", {}).get("metadata_output")


def execution_details(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_info": result.get("context_info"),
        "resource_metadata": result.get("resource_metadata"),
        "relationships": result.get("relationships"),
    }


def _write_upload_to_temp(file_name: str, file_bytes: bytes) -> str:
    suffix = Path(file_name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        return tmp.name


def _to_displayable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return json.loads(json.dumps(value, default=str))
