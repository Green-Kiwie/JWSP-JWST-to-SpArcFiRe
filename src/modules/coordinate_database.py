'''
Thread-safe SQLite database for managing galaxy coordinates and grouping.

This module provides fast coordinate lookups with spatial indexing to group
galaxies that share the same RA/DEC across multiple FITS files and wavebands.
'''

import sqlite3
import threading
import os
from typing import Optional, Tuple, List, Dict
import math


class CoordinateDatabase:
    '''Thread-safe database for galaxy coordinate management.
    
    Uses SQLite with R*Tree spatial indexing for fast coordinate lookups.
    Designed for concurrent access by multiple worker processes.
    '''
    
    def __init__(self, db_path: str, position_tolerance_arcsec: float = 1.0):
        '''Initialize coordinate database.
        
        Args:
            db_path: Path to SQLite database file
            position_tolerance_arcsec: Tolerance in arcseconds for matching coordinates (default: 1.0)
        '''
        self.db_path = db_path
        self.position_tolerance = position_tolerance_arcsec
        self._local = threading.local()
        
        # Initialize database schema
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        '''Get thread-local database connection.'''
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,  # Wait up to 30s for locks
                isolation_level=None  # Autocommit mode for better concurrency
            )
            # Set WAL mode only once when database is created
            try:
                self._local.conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass  # Already in WAL mode
            self._local.conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
        return self._local.conn
    
    def _init_database(self):
        '''Create database schema with spatial indexing.'''
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Main galaxy groups table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS galaxy_groups (
                group_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ra REAL NOT NULL,
                dec REAL NOT NULL,
                max_height INTEGER DEFAULT 0,
                max_width INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Index for coordinate lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_coordinates ON galaxy_groups(ra, dec)
        ''')
        
        # Individual galaxy instances (versions across different files/wavebands)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS galaxy_instances (
                instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                ra REAL NOT NULL,
                dec REAL NOT NULL,
                waveband TEXT NOT NULL,
                source_file TEXT NOT NULL,
                height INTEGER NOT NULL,
                width INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (group_id) REFERENCES galaxy_groups(group_id)
            )
        ''')
        
        # Index for quick lookups by group
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_group_id ON galaxy_instances(group_id)
        ''')
        
        # Index for waveband queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_waveband ON galaxy_instances(waveband)
        ''')
        
        conn.commit()
    
    def _coords_match(self, ra1: float, dec1: float, ra2: float, dec2: float) -> bool:
        '''Check if two coordinates match within tolerance.
        
        Uses spherical distance calculation for accuracy.
        '''
        # Convert tolerance from arcseconds to degrees
        tol_deg = self.position_tolerance / 3600.0
        
        # Simple box check first (fast)
        if abs(ra1 - ra2) > tol_deg or abs(dec1 - dec2) > tol_deg:
            return False
        
        # Spherical distance for accuracy (accounting for DEC scaling)
        # Distance = sqrt((ΔRA * cos(DEC))^2 + ΔDEC^2)
        avg_dec_rad = math.radians((dec1 + dec2) / 2.0)
        delta_ra = (ra1 - ra2) * math.cos(avg_dec_rad)
        delta_dec = dec1 - dec2
        distance_deg = math.sqrt(delta_ra**2 + delta_dec**2)
        
        return distance_deg <= tol_deg
    
    def find_matching_group(self, ra: float, dec: float) -> Optional[int]:
        '''Find existing group that matches given coordinates.
        
        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
            
        Returns:
            group_id if match found, None otherwise
        '''
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Search within a box around the target coordinates
        tol_deg = self.position_tolerance / 3600.0
        
        cursor.execute('''
            SELECT group_id, ra, dec
            FROM galaxy_groups
            WHERE ra BETWEEN ? AND ?
              AND dec BETWEEN ? AND ?
        ''', (ra - tol_deg, ra + tol_deg, dec - tol_deg, dec + tol_deg))
        
        # Check each candidate with precise spherical distance
        for group_id, group_ra, group_dec in cursor.fetchall():
            if self._coords_match(ra, dec, group_ra, group_dec):
                return group_id
        
        return None
    
    def add_galaxy(self, ra: float, dec: float, height: int, width: int, 
                   waveband: str, source_file: str) -> Tuple[int, int, float, float]:
        """
        Add a galaxy to a group (create new or add to existing).
        Returns: (group_id, version, canonical_ra, canonical_dec)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Find or create group
        group_id = self.find_matching_group(ra, dec)
        
        if group_id is None:
            # Create new group
            cursor.execute('''
                INSERT INTO galaxy_groups (ra, dec, max_height, max_width)
                VALUES (?, ?, ?, ?)
            ''', (ra, dec, height, width))
            group_id = cursor.lastrowid
            canonical_ra, canonical_dec = ra, dec
        else:
            # Update max dimensions if needed
            cursor.execute('''
                UPDATE galaxy_groups
                SET max_height = MAX(max_height, ?),
                    max_width = MAX(max_width, ?)
                WHERE group_id = ?
            ''', (height, width, group_id))
            # Retrieve canonical RA/DEC for the existing group
            cursor.execute('''
                SELECT ra, dec FROM galaxy_groups WHERE group_id = ?
            ''', (group_id,))
            canonical_ra, canonical_dec = cursor.fetchone()
        
        # Add instance
        cursor.execute('''
            INSERT INTO galaxy_instances 
            (group_id, ra, dec, waveband, source_file, height, width)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (group_id, ra, dec, waveband, source_file, height, width))
        
        # Get version number (count of instances in this group for this waveband)
        cursor.execute('''
            SELECT COUNT(*) FROM galaxy_instances
            WHERE group_id = ?
        ''', (group_id,))
        version = cursor.fetchone()[0]
        
        conn.commit()
        
        return group_id, version, canonical_ra, canonical_dec
    
    def find_galaxy_version_by_source(self, ra: float, dec: float, source_file: str, pos_tolerance: float = 0.5 / 3600.0) -> Optional[Tuple[int, int, float, float]]:
        """
        Find a specific galaxy instance by coordinates and source file.
        Returns (group_id, version, canonical_ra, canonical_dec) if found, None otherwise.
        
        The version is determined by counting how many instances in this group 
        come BEFORE or AT this instance's instance_id.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Find the group for this position
        cursor.execute('''
            SELECT group_id FROM galaxy_groups
            WHERE ABS(ra - ?) < ? AND ABS(dec - ?) < ?
            LIMIT 1
        ''', (ra, pos_tolerance, dec, pos_tolerance))
        
        result = cursor.fetchone()
        if not result:
            return None
        
        group_id = result[0]
        
        # Find this specific instance by source file
        cursor.execute('''
            SELECT instance_id FROM galaxy_instances
            WHERE group_id = ? AND source_file = ?
            LIMIT 1
        ''', (group_id, source_file))
        
        instance_result = cursor.fetchone()
        if not instance_result:
            return None
        
        instance_id = instance_result[0]
        
        # Get version: count instances in this group with instance_id <= current instance_id
        cursor.execute('''
            SELECT COUNT(*) FROM galaxy_instances
            WHERE group_id = ? AND instance_id <= ?
        ''', (group_id, instance_id))
        version = cursor.fetchone()[0]
        
        # Get canonical coordinates
        cursor.execute('''
            SELECT ra, dec FROM galaxy_groups WHERE group_id = ?
        ''', (group_id,))
        canonical_ra, canonical_dec = cursor.fetchone()
        
        return group_id, version, canonical_ra, canonical_dec

    def get_group_info(self, group_id: int) -> Optional[Dict]:
        """Get galaxy group information"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT group_id, ra, dec, max_height, max_width
            FROM galaxy_groups
            WHERE group_id = ?
        ''', (group_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            'group_id': row[0],
            'ra': row[1],
            'dec': row[2],
            'max_height': row[3],
            'max_width': row[4]
        }

    def get_group_instances(self, group_id: int) -> List[Dict]:
        '''Get all instances (versions) of a galaxy group.'''
        conn = self._get_connection()
        cursor = conn.cursor()

        
        cursor.execute('''
            SELECT instance_id, ra, dec, waveband, source_file, height, width
            FROM galaxy_instances
            WHERE group_id = ?
            ORDER BY instance_id
        ''', (group_id,))
        
        instances = []
        for row in cursor.fetchall():
            instances.append({
                'instance_id': row[0],
                'ra': row[1],
                'dec': row[2],
                'waveband': row[3],
                'source_file': row[4],
                'height': row[5],
                'width': row[6]
            })
        
        return instances
    
    def get_stats(self) -> Dict:
        '''Get database statistics.'''
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM galaxy_groups')
        group_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM galaxy_instances')
        instance_count = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(DISTINCT source_file) FROM galaxy_instances
        ''')
        file_count = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(DISTINCT waveband) FROM galaxy_instances
        ''')
        waveband_count = cursor.fetchone()[0]
        
        return {
            'total_groups': group_count,
            'total_instances': instance_count,
            'files_processed': file_count,
            'unique_wavebands': waveband_count,
            'avg_instances_per_group': instance_count / group_count if group_count > 0 else 0
        }
    
    def close(self):
        '''Close database connection for current thread.'''
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn
    
    def update_group_max_dimensions(self, group_id: int, max_height: int, max_width: int):
        '''Update the max dimensions for a group if larger sizes are found.'''
        conn = self._get_connection()
        conn.execute(
            '''UPDATE galaxy_groups 
               SET max_height = ?, max_width = ? 
               WHERE group_id = ?''',
            (max_height, max_width, group_id)
        )
        conn.commit()
