"""
SQLite Database Handler for Voice Inventory System

This module provides database operations for storing and retrieving
inventory entries, products, and transcription logs.
Includes connection pooling and proper error handling.
"""

import sqlite3
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import threading
import random
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Simple connection pool for SQLite.
    
    Manages a pool of database connections to avoid
    opening/closing connections for each operation.
    """
    
    def __init__(self, database_path: str, max_connections: int = 10):
        self.database_path = database_path
        self.max_connections = max_connections
        self._pool: List[sqlite3.Connection] = []
        self._lock = threading.Lock()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool or create a new one."""
        with self._lock:
            if self._pool:
                return self._pool.pop()
            else:
                conn = sqlite3.connect(
                    self.database_path,
                    check_same_thread=False,
                    timeout=30.0
                )
                conn.row_factory = sqlite3.Row
                return conn
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        with self._lock:
            if len(self._pool) < self.max_connections:
                self._pool.append(conn)
            else:
                conn.close()
    
    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()


class InventoryDatabase:
    """
    Database handler for the Voice Inventory System.
    
    Manages two main tables:
    - products: Master list of valid product codes
    - inventory_entries: Voice-captured inventory entries
    
    Includes connection pooling, indexing, and comprehensive
    error handling for production use.
    
    Attributes:
        database_path: Path to the SQLite database file
        pool: Connection pool for managing database connections
    
    Example:
        >>> db = InventoryDatabase('inventory.db')
        >>> db.initialize_database()
        >>> db.seed_sample_products()
        >>> db.insert_entry('ABC-123', 50, 'A-12', 0.95, 'Product ABC-123...')
    """
    
    def __init__(
        self,
        database_path: str = 'inventory.db',
        max_connections: int = 10
    ):
        """
        Initialize the InventoryDatabase.
        
        Args:
            database_path: Path to the SQLite database file
            max_connections: Maximum number of pooled connections
        """
        self.database_path = database_path
        self.pool = ConnectionPool(database_path, max_connections)
        
        # Ensure database directory exists
        db_dir = os.path.dirname(database_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        logger.info(f"InventoryDatabase initialized: {database_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = self.pool.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.pool.return_connection(conn)
    
    def initialize_database(self):
        """
        Create database tables and indexes if they don't exist.
        
        Creates:
        - products table: Master list of valid products
        - inventory_entries table: Voice-captured entries
        - failure_logs table: Failed transcription attempts
        - Indexes for performance optimization
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create products table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_code TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT,
                    description TEXT,
                    active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create inventory_entries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inventory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_code TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    location TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    raw_transcription TEXT,
                    audio_path TEXT,
                    verified BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_code) REFERENCES products(product_code)
                )
            ''')
            
            # Create failure_logs table for debugging
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS failure_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_path TEXT,
                    raw_transcription TEXT,
                    error_type TEXT NOT NULL,
                    error_message TEXT,
                    confidence_score REAL,
                    attempt_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_product_code 
                ON inventory_entries(product_code)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_location 
                ON inventory_entries(location)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON inventory_entries(created_at)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_products_code 
                ON products(product_code)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_products_active 
                ON products(active)
            ''')
            
            logger.info("Database tables and indexes created successfully")
    
    def insert_entry(
        self,
        product_code: str,
        quantity: int,
        location: str,
        confidence: float,
        transcription: str,
        audio_path: Optional[str] = None
    ) -> int:
        """
        Insert a new inventory entry.
        
        Args:
            product_code: Product code (e.g., "ABC-123")
            quantity: Quantity count
            location: Warehouse location (e.g., "A-12")
            confidence: Confidence score from speech recognition (0-1)
            transcription: Raw transcription text
            audio_path: Optional path to the audio file
        
        Returns:
            int: ID of the inserted entry
        
        Raises:
            ValueError: If required fields are missing or invalid
            sqlite3.Error: If database operation fails
        """
        if not product_code or not location:
            raise ValueError("Product code and location are required")
        
        if quantity < 1 or quantity > 9999:
            raise ValueError(f"Quantity must be between 1 and 9999, got {quantity}")
        
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO inventory_entries 
                (product_code, quantity, location, confidence_score, 
                 raw_transcription, audio_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (product_code, quantity, location, confidence, 
                  transcription, audio_path))
            
            entry_id = cursor.lastrowid
            logger.info(
                f"Inserted entry {entry_id}: {product_code} x{quantity} @ {location}"
            )
            
            return entry_id
    
    def product_exists(self, product_code: str) -> bool:
        """
        Check if a product exists in the database.
        
        Args:
            product_code: Product code to check
        
        Returns:
            bool: True if product exists and is active
        """
        if not product_code:
            return False
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM products 
                WHERE product_code = ? AND active = 1
            ''', (product_code.upper(),))
            
            count = cursor.fetchone()[0]
            return count > 0
    
    def get_product_details(self, product_code: str) -> Optional[Dict[str, Any]]:
        """
        Get product details by product code.
        
        Args:
            product_code: Product code to look up
        
        Returns:
            Dict with product details or None if not found
        """
        if not product_code:
            return None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, product_code, name, category, description, active,
                       created_at, updated_at
                FROM products 
                WHERE product_code = ?
            ''', (product_code.upper(),))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row['id'],
                    'product_code': row['product_code'],
                    'name': row['name'],
                    'category': row['category'],
                    'description': row['description'],
                    'active': bool(row['active']),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }
            
            return None
    
    def get_all_product_codes(self, active_only: bool = True) -> List[str]:
        """
        Get all product codes from the database.
        
        Args:
            active_only: If True, only return active products
        
        Returns:
            List of product codes
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if active_only:
                cursor.execute('''
                    SELECT product_code FROM products 
                    WHERE active = 1
                    ORDER BY product_code
                ''')
            else:
                cursor.execute('''
                    SELECT product_code FROM products 
                    ORDER BY product_code
                ''')
            
            return [row['product_code'] for row in cursor.fetchall()]
    
    def get_inventory_by_location(
        self,
        location: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get inventory entries by location.
        
        Args:
            location: Location code (e.g., "A-12")
            limit: Maximum number of results
        
        Returns:
            List of inventory entry dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT ie.*, p.name as product_name, p.category
                FROM inventory_entries ie
                LEFT JOIN products p ON ie.product_code = p.product_code
                WHERE ie.location = ?
                ORDER BY ie.created_at DESC
                LIMIT ?
            ''', (location.upper(), limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_inventory_entries(
        self,
        product_code: Optional[str] = None,
        location: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get inventory entries with optional filters.
        
        Args:
            product_code: Filter by product code
            location: Filter by location
            start_date: Filter entries after this date
            end_date: Filter entries before this date
            limit: Maximum number of results
            offset: Number of results to skip (for pagination)
        
        Returns:
            Tuple of (list of entries, total count)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query dynamically
            conditions = []
            params = []
            
            if product_code:
                conditions.append("ie.product_code = ?")
                params.append(product_code.upper())
            
            if location:
                conditions.append("ie.location = ?")
                params.append(location.upper())
            
            if start_date:
                conditions.append("ie.created_at >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                conditions.append("ie.created_at <= ?")
                params.append(end_date.isoformat())
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            # Get total count
            count_query = f'''
                SELECT COUNT(*) FROM inventory_entries ie
                {where_clause}
            '''
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()[0]
            
            # Get entries
            query = f'''
                SELECT ie.*, p.name as product_name, p.category
                FROM inventory_entries ie
                LEFT JOIN products p ON ie.product_code = p.product_code
                {where_clause}
                ORDER BY ie.created_at DESC
                LIMIT ? OFFSET ?
            '''
            cursor.execute(query, params + [limit, offset])
            
            entries = [dict(row) for row in cursor.fetchall()]
            
            return entries, total_count
    
    def add_product(
        self,
        product_code: str,
        name: str,
        category: Optional[str] = None,
        description: Optional[str] = None
    ) -> int:
        """
        Add a new product to the database.
        
        Args:
            product_code: Unique product code (e.g., "ABC-123")
            name: Product name
            category: Optional category
            description: Optional description
        
        Returns:
            int: ID of the inserted product
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO products (product_code, name, category, description)
                VALUES (?, ?, ?, ?)
            ''', (product_code.upper(), name, category, description))
            
            product_id = cursor.lastrowid
            logger.info(f"Added product: {product_code} - {name}")
            
            return product_id
    
    def seed_sample_products(self, count: int = 50) -> int:
        """
        Populate database with sample products for testing.
        
        Creates products with codes like "ABC-123", "XY-45678", etc.
        
        Args:
            count: Number of products to create (default 50)
        
        Returns:
            int: Number of products created
        """
        categories = [
            'Electronics', 'Hardware', 'Tools', 'Parts',
            'Supplies', 'Equipment', 'Materials', 'Components'
        ]
        
        prefixes = [
            'ABC', 'XY', 'DEF', 'GH', 'IJK', 'LM', 'NOP', 'QR',
            'STU', 'VW', 'XYZ', 'EL', 'HW', 'TL', 'PT', 'SP'
        ]
        
        items = [
            'Widget', 'Gadget', 'Component', 'Module', 'Assembly',
            'Unit', 'Part', 'Piece', 'Element', 'Device'
        ]
        
        created = 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for i in range(count):
                # Generate product code
                prefix = random.choice(prefixes)
                num_digits = random.choice([3, 4, 5])
                number = ''.join(random.choices('0123456789', k=num_digits))
                product_code = f"{prefix}-{number}"
                
                # Generate name
                item = random.choice(items)
                name = f"{item} {product_code}"
                
                # Random category
                category = random.choice(categories)
                
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO products 
                        (product_code, name, category)
                        VALUES (?, ?, ?)
                    ''', (product_code, name, category))
                    
                    if cursor.rowcount > 0:
                        created += 1
                except sqlite3.IntegrityError:
                    # Product code already exists, skip
                    continue
        
        logger.info(f"Seeded {created} sample products")
        return created
    
    def log_failure(
        self,
        error_type: str,
        error_message: str,
        audio_path: Optional[str] = None,
        transcription: Optional[str] = None,
        confidence: Optional[float] = None,
        attempt_count: int = 1
    ) -> int:
        """
        Log a failed transcription attempt for debugging.
        
        Args:
            error_type: Type of error (e.g., 'low_confidence', 'incomplete')
            error_message: Detailed error message
            audio_path: Path to the audio file
            transcription: Raw transcription if available
            confidence: Confidence score if available
            attempt_count: Number of attempts made
        
        Returns:
            int: ID of the log entry
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO failure_logs 
                (audio_path, raw_transcription, error_type, error_message,
                 confidence_score, attempt_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (audio_path, transcription, error_type, error_message,
                  confidence, attempt_count))
            
            log_id = cursor.lastrowid
            logger.warning(f"Logged failure {log_id}: {error_type} - {error_message}")
            
            return log_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics for monitoring.
        
        Returns:
            Dict with various statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute('SELECT COUNT(*) FROM inventory_entries')
            total_entries = cursor.fetchone()[0]
            
            # Total products
            cursor.execute('SELECT COUNT(*) FROM products WHERE active = 1')
            total_products = cursor.fetchone()[0]
            
            # Average confidence
            cursor.execute('SELECT AVG(confidence_score) FROM inventory_entries')
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Entries today
            today = datetime.now().date().isoformat()
            cursor.execute('''
                SELECT COUNT(*) FROM inventory_entries 
                WHERE date(created_at) = date(?)
            ''', (today,))
            entries_today = cursor.fetchone()[0]
            
            # Failure count
            cursor.execute('SELECT COUNT(*) FROM failure_logs')
            failures = cursor.fetchone()[0]
            
            # Success rate
            total_attempts = total_entries + failures
            success_rate = total_entries / total_attempts if total_attempts > 0 else 1.0
            
            return {
                'total_entries': total_entries,
                'total_products': total_products,
                'average_confidence': round(avg_confidence, 4),
                'entries_today': entries_today,
                'total_failures': failures,
                'success_rate': round(success_rate, 4),
                'database_path': self.database_path
            }
    
    def close(self):
        """Close all database connections."""
        self.pool.close_all()
        logger.info("Database connections closed")


# Default database instance
_default_db: Optional[InventoryDatabase] = None


def get_database(database_path: str = 'inventory.db') -> InventoryDatabase:
    """
    Get or create the default database instance.
    
    Args:
        database_path: Path to the database file
    
    Returns:
        InventoryDatabase instance
    """
    global _default_db
    
    if _default_db is None:
        _default_db = InventoryDatabase(database_path)
        _default_db.initialize_database()
    
    return _default_db


if __name__ == "__main__":
    # Test the database
    print("Inventory Database Module")
    print("=" * 50)
    
    # Create test database
    db = InventoryDatabase('test_inventory.db')
    db.initialize_database()
    
    # Seed sample products
    print("\nSeeding sample products...")
    created = db.seed_sample_products(50)
    print(f"Created {created} products")
    
    # Get all product codes
    codes = db.get_all_product_codes()
    print(f"\nSample product codes: {codes[:5]}...")
    
    # Test product lookup
    if codes:
        test_code = codes[0]
        exists = db.product_exists(test_code)
        print(f"\nProduct {test_code} exists: {exists}")
        
        details = db.get_product_details(test_code)
        if details:
            print(f"Details: {details['name']} ({details['category']})")
    
    # Test entry insertion
    if codes:
        entry_id = db.insert_entry(
            product_code=codes[0],
            quantity=50,
            location='A-12',
            confidence=0.95,
            transcription='Product ABC-123 quantity 50 location A-12'
        )
        print(f"\nInserted entry ID: {entry_id}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    db.close()
    print("\nDatabase test complete")
