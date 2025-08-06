"""Database configuration and models for API key management."""

import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()

class DatabaseManager:
    """Manages SQLite database operations for API keys."""
    
    def __init__(self, db_path: str = "data/mlx_server.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_name TEXT NOT NULL UNIQUE,
                    key_hash TEXT NOT NULL UNIQUE,
                    key_prefix TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    rate_limit INTEGER DEFAULT 1000,
                    metadata TEXT,
                    created_by TEXT DEFAULT 'system'
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_key_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key_id INTEGER NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    response_status INTEGER,
                    processing_time_ms REAL,
                    FOREIGN KEY (api_key_id) REFERENCES api_keys (id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON api_key_usage(timestamp);
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def _hash_key(api_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def generate_api_key() -> tuple[str, str]:
        """Generate a new API key with prefix."""
        prefix = "mlx_"
        key_part = secrets.token_urlsafe(32)
        full_key = f"{prefix}{key_part}"
        return full_key, prefix
    
    def create_api_key(
        self, 
        name: str, 
        expires_days: Optional[int] = None,
        rate_limit: int = 1000,
        metadata: Optional[str] = None,
        created_by: str = "admin"
    ) -> Dict[str, Any]:
        """Create a new API key."""
        api_key, prefix = self.generate_api_key()
        key_hash = self._hash_key(api_key)
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        with self._get_connection() as conn:
            try:
                cursor = conn.execute("""
                    INSERT INTO api_keys (
                        key_name, key_hash, key_prefix, expires_at, 
                        rate_limit, metadata, created_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (name, key_hash, prefix, expires_at, rate_limit, metadata, created_by))
                
                key_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Created API key '{name}' with ID {key_id}")
                
                return {
                    "id": key_id,
                    "name": name,
                    "api_key": api_key,  # Only returned on creation
                    "prefix": prefix,
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    "rate_limit": rate_limit,
                    "created_at": datetime.utcnow().isoformat(),
                    "is_active": True,
                    "usage_count": 0,  # New key starts with 0 usage
                    "created_by": created_by,  # Include the created_by parameter
                    "last_used_at": None,  # New key hasn't been used yet
                    "metadata": metadata  # Include metadata
                }
                
            except sqlite3.IntegrityError as e:
                if "key_name" in str(e):
                    raise ValueError(f"API key name '{name}' already exists")
                raise ValueError(f"Failed to create API key: {e}")
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key and return key information if valid."""
        key_hash = self._hash_key(api_key)
        
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, key_name, key_prefix, is_active, expires_at, 
                       usage_count, rate_limit, last_used_at, created_at, metadata
                FROM api_keys 
                WHERE key_hash = ? AND is_active = 1
            """, (key_hash,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Check if key has expired
            if row['expires_at']:
                expires_at = datetime.fromisoformat(row['expires_at'])
                if datetime.utcnow() > expires_at:
                    logger.warning(f"Expired API key used: {row['key_name']}")
                    return None
            
            # Update last used timestamp and usage count
            conn.execute("""
                UPDATE api_keys 
                SET last_used_at = CURRENT_TIMESTAMP, usage_count = usage_count + 1
                WHERE id = ?
            """, (row['id'],))
            conn.commit()
            
            return dict(row)
    
    def list_api_keys(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """List all API keys (without the actual key values)."""
        query = """
            SELECT id, key_name, key_prefix, is_active, created_at, 
                   last_used_at, expires_at, usage_count, rate_limit, 
                   created_by, metadata
            FROM api_keys
        """
        
        params = []
        if not include_inactive:
            query += " WHERE is_active = 1"
        
        query += " ORDER BY created_at DESC"
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_api_key(self, key_id: int) -> Optional[Dict[str, Any]]:
        """Get API key details by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, key_name, key_prefix, is_active, created_at, 
                       last_used_at, expires_at, usage_count, rate_limit, 
                       created_by, metadata
                FROM api_keys 
                WHERE id = ?
            """, (key_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def deactivate_api_key(self, key_id: int) -> bool:
        """Deactivate an API key."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE api_keys SET is_active = 0 WHERE id = ?
            """, (key_id,))
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Deactivated API key with ID {key_id}")
                return True
            return False
    
    def delete_api_key(self, key_id: int) -> bool:
        """Permanently delete an API key."""
        with self._get_connection() as conn:
            # First delete usage records
            conn.execute("DELETE FROM api_key_usage WHERE api_key_id = ?", (key_id,))
            
            # Then delete the API key
            cursor = conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Deleted API key with ID {key_id}")
                return True
            return False
    
    def log_api_usage(
        self, 
        api_key_id: int, 
        endpoint: str, 
        method: str, 
        response_status: int,
        processing_time_ms: Optional[float] = None
    ):
        """Log API key usage."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO api_key_usage (
                    api_key_id, endpoint, method, response_status, processing_time_ms
                ) VALUES (?, ?, ?, ?, ?)
            """, (api_key_id, endpoint, method, response_status, processing_time_ms))
            conn.commit()
    
    def get_usage_stats(self, key_id: Optional[int] = None, days: int = 7) -> Dict[str, Any]:
        """Get usage statistics for API keys."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        base_query = """
            SELECT 
                ak.id, ak.key_name, ak.key_prefix,
                COUNT(aku.id) as request_count,
                AVG(aku.processing_time_ms) as avg_processing_time,
                MAX(aku.timestamp) as last_request
            FROM api_keys ak
            LEFT JOIN api_key_usage aku ON ak.id = aku.api_key_id 
                AND aku.timestamp >= ?
        """
        
        params = [since_date.isoformat()]
        
        if key_id:
            base_query += " WHERE ak.id = ?"
            params.append(key_id)
        
        base_query += " GROUP BY ak.id ORDER BY request_count DESC"
        
        with self._get_connection() as conn:
            cursor = conn.execute(base_query, params)
            return [dict(row) for row in cursor.fetchall()]

# Global database manager instance
db_manager = DatabaseManager()