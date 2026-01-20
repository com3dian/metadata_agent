"""
DataSource Base Classes and Models.

This module defines the abstract base class for all data sources and
common data models used across the system.

The DataSource abstraction provides a unified interface for:
- Single files (CSV, Parquet, JSON)
- Multiple related files (CSV collections)
- Databases (SQLite, PostgreSQL)
- Directories of files
- Future: APIs, cloud storage, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd


class SourceType(str, Enum):
    """Enumeration of supported data source types."""

    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    SQLITE = "sqlite"
    DIRECTORY = "directory"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    """Information about a single column in a table."""

    name: str
    dtype: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_reference: Optional[str] = None  # "table.column" format
    description: Optional[str] = None
    sample_values: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "foreign_key_reference": self.foreign_key_reference,
            "description": self.description,
            "sample_values": self.sample_values[:5],  # Limit for serialization
        }


@dataclass
class TableInfo:
    """Information about a single table in the data source."""

    name: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: List[ColumnInfo] = field(default_factory=list)
    primary_key: Optional[str] = None
    file_path: Optional[str] = None  # For file-based sources
    file_size_bytes: Optional[int] = None
    description: Optional[str] = None

    @property
    def column_names(self) -> List[str]:
        return [col.name for col in self.columns]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [col.to_dict() for col in self.columns],
            "primary_key": self.primary_key,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "description": self.description,
        }


@dataclass
class RelationshipInfo:
    """Information about a relationship between tables."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str  # "one-to-one", "one-to-many", "many-to-many"
    confidence: float = 0.0
    is_verified: bool = False  # True if from schema, False if discovered
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
            "relationship_type": self.relationship_type,
            "confidence": self.confidence,
            "is_verified": self.is_verified,
            "description": self.description,
        }


class DataSource(ABC):
    """
    Abstract base class for all data sources.

    This provides a unified interface for accessing data regardless of
    the underlying storage mechanism (files, databases, etc.).

    All concrete implementations must provide:
    - Table listing and metadata
    - Data reading capabilities
    - Schema information
    """

    def __init__(self, name: str = "dataset", description: Optional[str] = None):
        self._name = name
        self._description = description
        self._table_cache: Dict[str, TableInfo] = {}
        self._relationship_cache: Optional[List[RelationshipInfo]] = None

    # ===================================================================
    #  ABSTRACT PROPERTIES (Must be implemented by subclasses)
    # ===================================================================

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """Return the type of this data source."""
        pass

    @property
    @abstractmethod
    def tables(self) -> List[str]:
        """Return list of table names in this data source."""
        pass

    # ===================================================================
    #  ABSTRACT METHODS (Must be implemented by subclasses)
    # ===================================================================

    @abstractmethod
    def _load_table_info(self, table: str) -> TableInfo:
        """
        Load metadata for a specific table.
        Called internally - results are cached.
        """
        pass

    @abstractmethod
    def read_table(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Read a table into a pandas DataFrame.

        Args:
            table: Name of the table to read
            columns: Specific columns to read (None = all)
            nrows: Number of rows to read (None = all)
            **kwargs: Additional arguments for the reader

        Returns:
            DataFrame with the table data
        """
        pass

    @abstractmethod
    def iter_table(
        self, table: str, chunksize: int = 10000, **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over a table in chunks (for large tables).

        Args:
            table: Name of the table to read
            chunksize: Number of rows per chunk
            **kwargs: Additional arguments for the reader

        Yields:
            DataFrames with chunks of data
        """
        pass

    # ===================================================================
    #  CONCRETE PROPERTIES
    # ===================================================================

    @property
    def name(self) -> str:
        """Return the name of this data source."""
        return self._name

    @property
    def description(self) -> Optional[str]:
        """Return the description of this data source."""
        return self._description

    @property
    def is_multi_table(self) -> bool:
        """Return True if this data source has multiple tables."""
        return len(self.tables) > 1

    @property
    def primary_table(self) -> Optional[str]:
        """Return the primary/main table if designated, else first table."""
        return self.tables[0] if self.tables else None

    # ===================================================================
    #  CONCRETE METHODS
    # ===================================================================

    def get_table_info(self, table: str) -> TableInfo:
        """
        Get metadata for a specific table (cached).

        Args:
            table: Name of the table

        Returns:
            TableInfo with metadata about the table
        """
        if table not in self.tables:
            raise ValueError(f"Table '{table}' not found. Available: {self.tables}")

        if table not in self._table_cache:
            self._table_cache[table] = self._load_table_info(table)

        return self._table_cache[table]

    def get_all_table_info(self) -> Dict[str, TableInfo]:
        """Get metadata for all tables."""
        return {table: self.get_table_info(table) for table in self.tables}

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the full schema of this data source.

        Returns:
            Dictionary with source metadata, tables, and relationships
        """
        return {
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type.value,
            "is_multi_table": self.is_multi_table,
            "tables": {
                name: info.to_dict() for name, info in self.get_all_table_info().items()
            },
            "relationships": [r.to_dict() for r in self.get_relationships()],
        }

    def get_relationships(self) -> List[RelationshipInfo]:
        """
        Get known or discovered relationships between tables.

        Override in subclasses to provide schema-defined relationships
        (e.g., from SQLite foreign keys).

        Returns:
            List of RelationshipInfo objects
        """
        if self._relationship_cache is None:
            self._relationship_cache = self._discover_relationships()
        return self._relationship_cache

    def _discover_relationships(self) -> List[RelationshipInfo]:
        """
        Attempt to discover relationships between tables.
        Default implementation - can be overridden by subclasses.
        """
        # Default: no automatic discovery (subclasses can override)
        return []

    def get_column_values(
        self, table: str, column: str, limit: Optional[int] = None
    ) -> List[Any]:
        """Get unique values from a column."""
        df = self.read_table(table, columns=[column])
        values = df[column].dropna().unique().tolist()
        if limit:
            return values[:limit]
        return values

    def validate(self) -> bool:
        """
        Validate that the data source is accessible and properly configured.

        Returns:
            True if valid, raises exception otherwise
        """
        if not self.tables:
            raise ValueError("Data source has no tables")

        for table in self.tables:
            self.get_table_info(table)  # Will raise if table is inaccessible

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this data source to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type.value,
            "tables": self.tables,
            "is_multi_table": self.is_multi_table,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"tables={self.tables}, "
            f"source_type='{self.source_type.value}')"
        )

    def __str__(self) -> str:
        table_info = (
            f"{len(self.tables)} table(s)" if self.is_multi_table else self.tables[0]
        )
        return f"{self.name} ({self.source_type.value}: {table_info})"
