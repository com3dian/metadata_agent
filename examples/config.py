"""Configuration for example scripts.

These settings describe how an example run should read inputs, write outputs,
and display logs. Agent and LLM initialization settings live in `src.config`.
"""

import os

from dotenv import load_dotenv


load_dotenv()


# Dataset path used by generation examples.
# Can be overridden by environment variable: DATA_FILE
DATA_FILE = os.getenv("DATA_FILE")

# Directory for generated outputs.
# Can be overridden by environment variable: OUTPUT_DIR
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# Logging settings for examples.
# Can be overridden by environment variables: LOG_MODE, LOG_LEVEL
LOG_MODE = os.getenv("LOG_MODE", "self-contained")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def config_summary() -> str:
    """Return the active example script configuration."""
    return f"""
        Runtime Configuration:
        ----------------------
        Data file: {DATA_FILE or 'Not set'}
        Output directory: {OUTPUT_DIR}
        Log mode: {LOG_MODE}
        Log level: {LOG_LEVEL}
        """
