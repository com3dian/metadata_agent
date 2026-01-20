"""
DataSource Factory - Unified entry point for creating data sources.

This module provides automatic detection and creation of the appropriate
DataSource implementation based on the input provided.
"""
import os
from typing import Union, List, Dict, Optional, Any
from pathlib import Path
import glob

from .base import DataSource, SourceType
from .csv_source import CSVDataSource
from .sqlite_source import SQLiteDataSource


class DataSourceFactory:
    """
    Factory for creating DataSource instances with automatic type detection.
    
    This is the main entry point for creating data sources. It automatically
    detects the appropriate DataSource type based on the input format.
    
    Supported input formats:
    - str: Single file path, directory path, or connection string
    - List[str]: Multiple file paths
    - Dict[str, str]: Mapping of table names to file paths
    - DataSource: Pass-through (already a DataSource)
    
    Examples:
        # Single CSV file
        ds = DataSourceFactory.create("./data/users.csv")
        
        # Multiple CSV files
        ds = DataSourceFactory.create(["./data/users.csv", "./data/orders.csv"])
        
        # Named tables
        ds = DataSourceFactory.create({
            "users": "./data/users.csv",
            "orders": "./data/orders.csv"
        })
        
        # SQLite database
        ds = DataSourceFactory.create("./data/mydb.sqlite")
        
        # Directory of CSVs
        ds = DataSourceFactory.create("./data/")
        
        # Auto-detect with name
        ds = DataSourceFactory.create("./data/users.csv", name="user_data")
    """
    
    # File extensions mapped to source types
    EXTENSION_MAP = {
        '.csv': SourceType.CSV,
        '.tsv': SourceType.CSV,
        '.txt': SourceType.CSV,  # Might be delimited
        '.sqlite': SourceType.SQLITE,
        '.sqlite3': SourceType.SQLITE,
        '.db': SourceType.SQLITE,
        '.parquet': SourceType.PARQUET,
        '.json': SourceType.JSON,
        '.jsonl': SourceType.JSON,
    }
    
    @classmethod
    def create(
        cls,
        source: Union[str, List[str], Dict[str, str], DataSource],
        name: str = "dataset",
        description: Optional[str] = None,
        **kwargs
    ) -> DataSource:
        """
        Create a DataSource from various input formats.
        
        Args:
            source: The data source specification (see class docstring for formats)
            name: Name for the dataset
            description: Optional description
            **kwargs: Additional arguments passed to the specific DataSource
            
        Returns:
            Appropriate DataSource implementation
            
        Raises:
            ValueError: If the source format is not supported
            FileNotFoundError: If specified files/directories don't exist
        """
        # Already a DataSource - pass through
        if isinstance(source, DataSource):
            return source
        
        # String input - detect type
        if isinstance(source, str):
            return cls._create_from_string(source, name, description, **kwargs)
        
        # List of paths
        if isinstance(source, list):
            return cls._create_from_list(source, name, description, **kwargs)
        
        # Dict of table_name -> path
        if isinstance(source, dict):
            return cls._create_from_dict(source, name, description, **kwargs)
        
        raise ValueError(
            f"Cannot create DataSource from type: {type(source)}. "
            f"Supported: str, List[str], Dict[str, str], DataSource"
        )
    
    @classmethod
    def _create_from_string(
        cls,
        path: str,
        name: str,
        description: Optional[str],
        **kwargs
    ) -> DataSource:
        """Create DataSource from a string path."""
        path = os.path.expanduser(path)
        
        # Check if it's a directory
        if os.path.isdir(path):
            return cls._create_from_directory(path, name, description, **kwargs)
        
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        # Detect type from extension
        source_type = cls._detect_type_from_extension(path)
        
        return cls._create_typed_source(
            source_type, 
            path, 
            name, 
            description, 
            **kwargs
        )
    
    @classmethod
    def _create_from_list(
        cls,
        paths: List[str],
        name: str,
        description: Optional[str],
        **kwargs
    ) -> DataSource:
        """Create DataSource from a list of file paths."""
        if not paths:
            raise ValueError("Empty path list provided")
        
        # Expand and validate paths
        expanded_paths = []
        for p in paths:
            p = os.path.expanduser(p)
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}")
            expanded_paths.append(p)
        
        # Detect type from first file
        source_type = cls._detect_type_from_extension(expanded_paths[0])
        
        # For CSVs, create multi-file source
        if source_type == SourceType.CSV:
            # Convert to dict with derived names
            tables = {Path(p).stem: p for p in expanded_paths}
            return CSVDataSource(tables, name=name, description=description, **kwargs)
        
        raise ValueError(
            f"List of {source_type.value} files not supported. "
            f"Use a dict mapping or single file."
        )
    
    @classmethod
    def _create_from_dict(
        cls,
        tables: Dict[str, str],
        name: str,
        description: Optional[str],
        **kwargs
    ) -> DataSource:
        """Create DataSource from a dict of table_name -> path."""
        if not tables:
            raise ValueError("Empty tables dict provided")
        
        # Validate paths exist
        for table_name, path in tables.items():
            path = os.path.expanduser(path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found for table '{table_name}': {path}")
        
        # Detect type from first file
        first_path = list(tables.values())[0]
        source_type = cls._detect_type_from_extension(first_path)
        
        if source_type == SourceType.CSV:
            return CSVDataSource(tables, name=name, description=description, **kwargs)
        
        raise ValueError(
            f"Dict of {source_type.value} files not supported as multi-table source."
        )
    
    @classmethod
    def _create_from_directory(
        cls,
        dir_path: str,
        name: str,
        description: Optional[str],
        pattern: str = "*.csv",
        **kwargs
    ) -> DataSource:
        """Create DataSource from a directory of files."""
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        # Find matching files
        search_pattern = os.path.join(dir_path, pattern)
        files = glob.glob(search_pattern)
        
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {dir_path}"
            )
        
        # Create dict with derived names
        tables = {Path(f).stem: f for f in files}
        
        # Determine type from pattern/files
        if pattern.endswith('.csv') or pattern.endswith('.tsv'):
            return CSVDataSource(tables, name=name, description=description, **kwargs)
        
        # Default to CSV for unknown patterns
        return CSVDataSource(tables, name=name, description=description, **kwargs)
    
    @classmethod
    def _detect_type_from_extension(cls, path: str) -> SourceType:
        """Detect source type from file extension."""
        ext = Path(path).suffix.lower()
        return cls.EXTENSION_MAP.get(ext, SourceType.UNKNOWN)
    
    @classmethod
    def _create_typed_source(
        cls,
        source_type: SourceType,
        path: str,
        name: str,
        description: Optional[str],
        **kwargs
    ) -> DataSource:
        """Create a specific DataSource type."""
        if source_type == SourceType.CSV:
            return CSVDataSource(path, name=name, description=description, **kwargs)
        
        elif source_type == SourceType.SQLITE:
            return SQLiteDataSource(path, name=name, description=description, **kwargs)
        
        elif source_type == SourceType.PARQUET:
            raise NotImplementedError("Parquet support coming soon")
        
        elif source_type == SourceType.JSON:
            raise NotImplementedError("JSON support coming soon")
        
        else:
            # Try CSV as fallback
            return CSVDataSource(path, name=name, description=description, **kwargs)
    
    @classmethod
    def detect_type(cls, source: Union[str, List[str], Dict[str, str]]) -> SourceType:
        """
        Detect the source type without creating the DataSource.
        
        Useful for validation or UI hints.
        """
        if isinstance(source, str):
            if os.path.isdir(source):
                return SourceType.DIRECTORY
            return cls._detect_type_from_extension(source)
        
        elif isinstance(source, list) and source:
            return cls._detect_type_from_extension(source[0])
        
        elif isinstance(source, dict) and source:
            first_path = list(source.values())[0]
            return cls._detect_type_from_extension(first_path)
        
        return SourceType.UNKNOWN
    
    @classmethod
    def is_supported(cls, source: Union[str, List[str], Dict[str, str]]) -> bool:
        """Check if a source format is supported."""
        try:
            source_type = cls.detect_type(source)
            return source_type in [SourceType.CSV, SourceType.SQLITE, SourceType.DIRECTORY]
        except Exception:
            return False


# Convenience function
def create_datasource(
    source: Union[str, List[str], Dict[str, str], DataSource],
    name: str = "dataset",
    **kwargs
) -> DataSource:
    """
    Convenience function to create a DataSource.
    
    Equivalent to DataSourceFactory.create() but shorter to type.
    
    Examples:
        ds = create_datasource("./data/users.csv")
        ds = create_datasource(["./a.csv", "./b.csv"], name="my_data")
    """
    return DataSourceFactory.create(source, name=name, **kwargs)
