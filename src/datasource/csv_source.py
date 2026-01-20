"""
CSV DataSource Implementation.

Handles both single CSV files and collections of related CSV files.
Supports various CSV dialects including TSV (tab-separated).
"""
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Iterator, Union
from pathlib import Path

from .base import DataSource, SourceType, TableInfo, ColumnInfo, RelationshipInfo


class CSVDataSource(DataSource):
    """
    DataSource implementation for CSV files.
    
    Supports:
    - Single CSV file (treated as single-table dataset)
    - Multiple CSV files (treated as multi-table dataset)
    - Various delimiters (comma, tab, etc.)
    - Large file handling via chunked reading
    
    Examples:
        # Single file
        ds = CSVDataSource("./data/users.csv")
        
        # Multiple files with explicit names
        ds = CSVDataSource({
            "users": "./data/users.csv",
            "orders": "./data/orders.csv"
        })
        
        # List of files (names derived from filenames)
        ds = CSVDataSource(["./data/users.csv", "./data/orders.csv"])
    """
    
    # Threshold for considering a file "large" (50MB)
    LARGE_FILE_THRESHOLD = 50 * 1024 * 1024
    DEFAULT_CHUNK_SIZE = 10000
    
    def __init__(
        self,
        source: Union[str, List[str], Dict[str, str]],
        name: str = "dataset",
        description: Optional[str] = None,
        delimiter: Optional[str] = None,
        encoding: str = "utf-8",
        **read_options
    ):
        """
        Initialize a CSV DataSource.
        
        Args:
            source: File path, list of paths, or dict mapping table names to paths
            name: Name for this dataset
            description: Optional description
            delimiter: CSV delimiter (None = auto-detect)
            encoding: File encoding
            **read_options: Additional options passed to pandas read_csv
        """
        super().__init__(name=name, description=description)
        
        # Normalize source to dict of table_name -> path
        self._tables: Dict[str, str] = self._normalize_source(source)
        self._delimiter = delimiter
        self._encoding = encoding
        self._read_options = read_options
        
        # Cache for detected delimiters per file
        self._delimiter_cache: Dict[str, str] = {}
    
    def _normalize_source(
        self, 
        source: Union[str, List[str], Dict[str, str]]
    ) -> Dict[str, str]:
        """Convert various input formats to dict of table_name -> path."""
        if isinstance(source, str):
            # Single file path
            table_name = Path(source).stem
            return {table_name: os.path.abspath(source)}
        
        elif isinstance(source, list):
            # List of file paths
            return {
                Path(p).stem: os.path.abspath(p)
                for p in source
            }
        
        elif isinstance(source, dict):
            # Already a dict
            return {
                name: os.path.abspath(path)
                for name, path in source.items()
            }
        
        else:
            raise ValueError(f"Invalid source type: {type(source)}")
    
    def _detect_delimiter(self, file_path: str) -> str:
        """Auto-detect the delimiter used in a CSV file."""
        if file_path in self._delimiter_cache:
            return self._delimiter_cache[file_path]
        
        if self._delimiter:
            self._delimiter_cache[file_path] = self._delimiter
            return self._delimiter
        
        # Read first few lines and detect
        try:
            with open(file_path, 'r', encoding=self._encoding) as f:
                sample = f.read(8192)  # Read first 8KB
            
            # Count potential delimiters
            delimiters = [',', '\t', ';', '|']
            counts = {d: sample.count(d) for d in delimiters}
            
            # Pick the most common one that appears consistently
            best_delimiter = max(counts, key=counts.get)
            
            # Validate by checking if it creates consistent columns
            lines = sample.split('\n')[:5]
            if lines:
                col_counts = [len(line.split(best_delimiter)) for line in lines if line.strip()]
                if col_counts and len(set(col_counts)) == 1 and col_counts[0] > 1:
                    self._delimiter_cache[file_path] = best_delimiter
                    return best_delimiter
            
            # Default to comma
            self._delimiter_cache[file_path] = ','
            return ','
            
        except Exception:
            self._delimiter_cache[file_path] = ','
            return ','
    
    def _get_read_kwargs(self, table: str) -> Dict[str, Any]:
        """Get kwargs for pandas read_csv."""
        file_path = self._tables[table]
        delimiter = self._detect_delimiter(file_path)
        
        kwargs = {
            'delimiter': delimiter,
            'encoding': self._encoding,
            **self._read_options
        }
        return kwargs
    
    def _is_large_file(self, file_path: str) -> bool:
        """Check if file exceeds the large file threshold."""
        try:
            return os.path.getsize(file_path) > self.LARGE_FILE_THRESHOLD
        except OSError:
            return False
    
    # ===================================================================
    #  ABSTRACT METHOD IMPLEMENTATIONS
    # ===================================================================
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.CSV
    
    @property
    def tables(self) -> List[str]:
        return list(self._tables.keys())
    
    def _load_table_info(self, table: str) -> TableInfo:
        """Load metadata for a CSV file."""
        file_path = self._tables[table]
        kwargs = self._get_read_kwargs(table)
        
        # Get file size
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else None
        
        # Read sample to get column info
        sample_df = pd.read_csv(file_path, nrows=100, **kwargs)
        
        # Count total rows (efficiently for large files)
        if self._is_large_file(file_path):
            row_count = sum(
                len(chunk) 
                for chunk in pd.read_csv(
                    file_path, 
                    chunksize=self.DEFAULT_CHUNK_SIZE,
                    usecols=[0],
                    **kwargs
                )
            )
        else:
            row_count = len(pd.read_csv(file_path, usecols=[0], **kwargs))
        
        # Build column info
        columns = []
        for col in sample_df.columns:
            col_data = sample_df[col]
            columns.append(ColumnInfo(
                name=col,
                dtype=str(col_data.dtype),
                nullable=col_data.isnull().any(),
                sample_values=col_data.dropna().head(5).tolist()
            ))
        
        return TableInfo(
            name=table,
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
            file_path=file_path,
            file_size_bytes=file_size
        )
    
    def read_table(
        self, 
        table: str, 
        columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Read a CSV file into a DataFrame."""
        if table not in self._tables:
            raise ValueError(f"Table '{table}' not found. Available: {self.tables}")
        
        file_path = self._tables[table]
        read_kwargs = {**self._get_read_kwargs(table), **kwargs}
        
        if columns:
            read_kwargs['usecols'] = columns
        if nrows:
            read_kwargs['nrows'] = nrows
        
        return pd.read_csv(file_path, **read_kwargs)
    
    def iter_table(
        self, 
        table: str, 
        chunksize: int = 10000,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """Iterate over a CSV file in chunks."""
        if table not in self._tables:
            raise ValueError(f"Table '{table}' not found. Available: {self.tables}")
        
        file_path = self._tables[table]
        read_kwargs = {**self._get_read_kwargs(table), **kwargs}
        
        return pd.read_csv(file_path, chunksize=chunksize, **read_kwargs)
    
    # ===================================================================
    #  CSV-SPECIFIC METHODS
    # ===================================================================
    
    def get_file_path(self, table: str) -> str:
        """Get the file path for a specific table."""
        if table not in self._tables:
            raise ValueError(f"Table '{table}' not found. Available: {self.tables}")
        return self._tables[table]
    
    def get_all_file_paths(self) -> Dict[str, str]:
        """Get all table name to file path mappings."""
        return self._tables.copy()
    
    def get_delimiter(self, table: str) -> str:
        """Get the detected delimiter for a table."""
        file_path = self._tables[table]
        return self._detect_delimiter(file_path)
    
    def _discover_relationships(self) -> List[RelationshipInfo]:
        """
        Attempt to discover relationships between CSV tables.
        Uses column name matching and value overlap analysis.
        """
        if not self.is_multi_table:
            return []
        
        relationships = []
        tables_data: Dict[str, pd.DataFrame] = {}
        
        # Load sample data from each table
        for table in self.tables:
            try:
                tables_data[table] = self.read_table(table, nrows=1000)
            except Exception:
                continue
        
        # Compare each pair of tables
        table_list = list(tables_data.keys())
        for i, table_a in enumerate(table_list):
            for table_b in table_list[i+1:]:
                df_a = tables_data[table_a]
                df_b = tables_data[table_b]
                
                # Find columns with similar names
                for col_a in df_a.columns:
                    col_a_norm = col_a.lower().replace("_", "").replace("-", "")
                    
                    for col_b in df_b.columns:
                        col_b_norm = col_b.lower().replace("_", "").replace("-", "")
                        
                        # Check for name similarity
                        name_match = (
                            col_a_norm == col_b_norm or
                            (col_a_norm.endswith("id") and col_b_norm.endswith("id") and
                             col_a_norm[:-2] == col_b_norm[:-2]) or
                            col_a_norm in col_b_norm or col_b_norm in col_a_norm
                        )
                        
                        if not name_match:
                            continue
                        
                        # Analyze value overlap
                        values_a = set(df_a[col_a].dropna().unique())
                        values_b = set(df_b[col_b].dropna().unique())
                        
                        if not values_a or not values_b:
                            continue
                        
                        intersection = values_a & values_b
                        if not intersection:
                            continue
                        
                        match_rate = len(intersection) / min(len(values_a), len(values_b))
                        
                        if match_rate > 0.1:  # At least 10% overlap
                            # Determine relationship type
                            unique_ratio_a = len(values_a) / len(df_a)
                            unique_ratio_b = len(values_b) / len(df_b)
                            
                            if unique_ratio_a > 0.9 and unique_ratio_b > 0.9:
                                rel_type = "one-to-one"
                            elif unique_ratio_a > 0.9:
                                rel_type = "one-to-many"
                            elif unique_ratio_b > 0.9:
                                rel_type = "many-to-one"
                            else:
                                rel_type = "many-to-many"
                            
                            confidence = match_rate * 0.8 + 0.2  # Boost for name match
                            
                            relationships.append(RelationshipInfo(
                                from_table=table_a,
                                from_column=col_a,
                                to_table=table_b,
                                to_column=col_b,
                                relationship_type=rel_type,
                                confidence=round(confidence, 3),
                                is_verified=False
                            ))
        
        # Sort by confidence
        relationships.sort(key=lambda r: r.confidence, reverse=True)
        return relationships
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with CSV-specific info."""
        base = super().to_dict()
        base["file_paths"] = self._tables
        base["delimiters"] = {
            table: self.get_delimiter(table) 
            for table in self.tables
        }
        return base
