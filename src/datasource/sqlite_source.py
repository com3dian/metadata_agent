"""
SQLite DataSource Implementation.

Handles SQLite database files with multiple tables.
Extracts schema information including foreign key relationships.
"""
import os
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

from .base import DataSource, SourceType, TableInfo, ColumnInfo, RelationshipInfo


class SQLiteDataSource(DataSource):
    """
    DataSource implementation for SQLite databases.
    
    Features:
    - Automatic table discovery
    - Schema extraction from database metadata
    - Foreign key relationship detection
    - Efficient chunked reading for large tables
    
    Examples:
        # From file path
        ds = SQLiteDataSource("./data/mydb.sqlite")
        
        # With custom name
        ds = SQLiteDataSource("./data/mydb.db", name="my_database")
    """
    
    DEFAULT_CHUNK_SIZE = 10000
    
    def __init__(
        self,
        db_path: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        exclude_tables: Optional[List[str]] = None
    ):
        """
        Initialize a SQLite DataSource.
        
        Args:
            db_path: Path to the SQLite database file
            name: Name for this dataset (defaults to filename)
            description: Optional description
            exclude_tables: List of table names to exclude (e.g., system tables)
        """
        self._db_path = os.path.abspath(db_path)
        
        # Default name from filename
        if name is None:
            name = Path(db_path).stem
        
        super().__init__(name=name, description=description)
        
        self._exclude_tables = set(exclude_tables or [])
        self._exclude_tables.update(['sqlite_sequence', 'sqlite_stat1'])  # System tables
        
        # Cache for table list
        self._tables_cache: Optional[List[str]] = None
    
    def _get_connection(self) -> sqlite3.Connection:
        """Create a database connection."""
        if not os.path.exists(self._db_path):
            raise FileNotFoundError(f"Database not found: {self._db_path}")
        return sqlite3.connect(self._db_path)
    
    # ===================================================================
    #  ABSTRACT METHOD IMPLEMENTATIONS
    # ===================================================================
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.SQLITE
    
    @property
    def tables(self) -> List[str]:
        """Get list of tables in the database."""
        if self._tables_cache is not None:
            return self._tables_cache
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            all_tables = [row[0] for row in cursor.fetchall()]
        
        # Filter out excluded tables
        self._tables_cache = [t for t in all_tables if t not in self._exclude_tables]
        return self._tables_cache
    
    def _load_table_info(self, table: str) -> TableInfo:
        """Load metadata for a database table."""
        with self._get_connection() as conn:
            # Get column info using PRAGMA
            cursor = conn.execute(f"PRAGMA table_info('{table}')")
            pragma_info = cursor.fetchall()
            
            # Get row count
            cursor = conn.execute(f"SELECT COUNT(*) FROM '{table}'")
            row_count = cursor.fetchone()[0]
            
            # Get primary key columns
            pk_columns = {row[1] for row in pragma_info if row[5] > 0}  # pk column is index 5
            
            # Get foreign key info
            cursor = conn.execute(f"PRAGMA foreign_key_list('{table}')")
            fk_info = cursor.fetchall()
            fk_columns = {row[3]: f"{row[2]}.{row[4]}" for row in fk_info}
            
            # Get sample data for column values
            sample_df = pd.read_sql_query(
                f"SELECT * FROM '{table}' LIMIT 100", conn
            )
            
            # Build column info
            columns = []
            for row in pragma_info:
                col_name = row[1]
                col_type = row[2]
                not_null = bool(row[3])
                
                sample_values = []
                if col_name in sample_df.columns:
                    sample_values = sample_df[col_name].dropna().head(5).tolist()
                
                columns.append(ColumnInfo(
                    name=col_name,
                    dtype=col_type,
                    nullable=not not_null,
                    is_primary_key=col_name in pk_columns,
                    is_foreign_key=col_name in fk_columns,
                    foreign_key_reference=fk_columns.get(col_name),
                    sample_values=sample_values
                ))
            
            # Determine primary key
            primary_key = None
            if pk_columns:
                primary_key = list(pk_columns)[0] if len(pk_columns) == 1 else list(pk_columns)
        
        return TableInfo(
            name=table,
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
            primary_key=primary_key,
            file_path=self._db_path
        )
    
    def read_table(
        self, 
        table: str, 
        columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Read a table into a DataFrame."""
        if table not in self.tables:
            raise ValueError(f"Table '{table}' not found. Available: {self.tables}")
        
        # Build query
        col_str = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        query = f'SELECT {col_str} FROM "{table}"'
        
        if nrows:
            query += f" LIMIT {nrows}"
        
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, **kwargs)
    
    def iter_table(
        self, 
        table: str, 
        chunksize: int = 10000,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """Iterate over a table in chunks."""
        if table not in self.tables:
            raise ValueError(f"Table '{table}' not found. Available: {self.tables}")
        
        query = f'SELECT * FROM "{table}"'
        
        with self._get_connection() as conn:
            for chunk in pd.read_sql_query(query, conn, chunksize=chunksize, **kwargs):
                yield chunk
    
    # ===================================================================
    #  SQLITE-SPECIFIC METHODS
    # ===================================================================
    
    def get_db_path(self) -> str:
        """Get the database file path."""
        return self._db_path
    
    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute an arbitrary SQL query and return results as DataFrame."""
        with self._get_connection() as conn:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            return pd.read_sql_query(query, conn)
    
    def _discover_relationships(self) -> List[RelationshipInfo]:
        """
        Extract foreign key relationships from SQLite schema.
        
        SQLite stores FK info that we can query directly,
        providing verified relationships.
        """
        relationships = []
        
        with self._get_connection() as conn:
            for table in self.tables:
                cursor = conn.execute(f"PRAGMA foreign_key_list('{table}')")
                fk_rows = cursor.fetchall()
                
                for fk in fk_rows:
                    # fk format: (id, seq, table, from, to, on_update, on_delete, match)
                    to_table = fk[2]
                    from_col = fk[3]
                    to_col = fk[4]
                    
                    relationships.append(RelationshipInfo(
                        from_table=table,
                        from_column=from_col,
                        to_table=to_table,
                        to_column=to_col,
                        relationship_type="many-to-one",  # FK typically implies many-to-one
                        confidence=1.0,  # Schema-defined = 100% confidence
                        is_verified=True,
                        description=f"Foreign key constraint from {table}.{from_col}"
                    ))
        
        return relationships
    
    def get_table_ddl(self, table: str) -> str:
        """Get the CREATE TABLE statement for a table."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            )
            result = cursor.fetchone()
            return result[0] if result else ""
    
    def get_indexes(self, table: str) -> List[Dict[str, Any]]:
        """Get index information for a table."""
        with self._get_connection() as conn:
            cursor = conn.execute(f"PRAGMA index_list('{table}')")
            indexes = []
            for idx_row in cursor.fetchall():
                idx_name = idx_row[1]
                cursor2 = conn.execute(f"PRAGMA index_info('{idx_name}')")
                columns = [row[2] for row in cursor2.fetchall()]
                indexes.append({
                    "name": idx_name,
                    "unique": bool(idx_row[2]),
                    "columns": columns
                })
            return indexes
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with SQLite-specific info."""
        base = super().to_dict()
        base["db_path"] = self._db_path
        base["file_size_bytes"] = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else None
        return base
