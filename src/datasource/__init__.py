"""
DataSource Module - Unified data access layer.

This module provides a unified interface for accessing data from various
sources including CSV files, SQLite databases, and more.

The DataSource abstraction allows the rest of the system to work with
data regardless of the underlying storage mechanism.

Quick Start:
    from src.datasource import create_datasource
    
    # Single CSV file
    ds = create_datasource("./data/users.csv")
    
    # Multiple related CSVs
    ds = create_datasource({
        "users": "./data/users.csv",
        "orders": "./data/orders.csv"
    })
    
    # SQLite database
    ds = create_datasource("./data/mydb.sqlite")
    
    # Directory of CSVs
    ds = create_datasource("./data/", pattern="*.csv")
    
    # Use the DataSource
    print(ds.tables)  # ['users', 'orders']
    df = ds.read_table("users")
    info = ds.get_table_info("users")
    schema = ds.get_schema()
"""

from .base import (
    DataSource,
    SourceType,
    TableInfo,
    ColumnInfo,
    RelationshipInfo
)
from .csv_source import CSVDataSource
from .sqlite_source import SQLiteDataSource
from .factory import DataSourceFactory, create_datasource

__all__ = [
    # Base classes and models
    "DataSource",
    "SourceType",
    "TableInfo",
    "ColumnInfo",
    "RelationshipInfo",
    # Concrete implementations
    "CSVDataSource",
    "SQLiteDataSource",
    # Factory
    "DataSourceFactory",
    "create_datasource",
]
