"""History database for tracking walked segments."""

import json
import sqlite3
from datetime import datetime
from typing import Optional


class HistoryDB:
    """SQLite database for tracking walked segments"""

    def __init__(self, db_path: str = "walker_history.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        """Create database tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS segment_history (
                segment_id TEXT PRIMARY KEY,
                times_walked INTEGER DEFAULT 0,
                last_walked TEXT,
                first_walked TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS walks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                ended_at TEXT,
                distance_meters REAL,
                segments_walked INTEGER
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS exclusion_zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL DEFAULT 'Unnamed Zone',
                polygon TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def record_segment(self, segment_id: str):
        """Record that a segment was walked"""
        now = datetime.now().isoformat()
        self.conn.execute("""
            INSERT INTO segment_history (segment_id, times_walked, last_walked, first_walked)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(segment_id) DO UPDATE SET
                times_walked = times_walked + 1,
                last_walked = ?
        """, (segment_id, now, now, now))
        self.conn.commit()

    def get_segment_history(self, segment_id: str) -> tuple[int, Optional[str]]:
        """Get (times_walked, last_walked) for a segment"""
        cursor = self.conn.execute(
            "SELECT times_walked, last_walked FROM segment_history WHERE segment_id = ?",
            (segment_id,)
        )
        row = cursor.fetchone()
        if row:
            return row[0], row[1]
        return 0, None

    def start_walk(self) -> int:
        """Start a new walk, return walk ID"""
        now = datetime.now().isoformat()
        cursor = self.conn.execute(
            "INSERT INTO walks (started_at) VALUES (?)",
            (now,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def end_walk(self, walk_id: int, distance: float, segments: int):
        """End a walk with stats"""
        now = datetime.now().isoformat()
        self.conn.execute(
            "UPDATE walks SET ended_at = ?, distance_meters = ?, segments_walked = ? WHERE id = ?",
            (now, distance, segments, walk_id)
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        """Get overall walking stats"""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_walks,
                SUM(distance_meters) as total_distance,
                COUNT(DISTINCT segment_id) as unique_segments
            FROM walks, segment_history
        """)
        row = cursor.fetchone()
        return {
            "total_walks": row[0] or 0,
            "total_distance_km": (row[1] or 0) / 1000,
            "unique_segments": row[2] or 0
        }

    def add_exclusion_zone(self, name: str, polygon: list[list[float]]) -> int:
        """Add an exclusion zone. Returns the zone ID."""
        now = datetime.now().isoformat()
        cursor = self.conn.execute(
            "INSERT INTO exclusion_zones (name, polygon, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (name, json.dumps(polygon), now, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_exclusion_zones(self) -> list[dict]:
        """Get all exclusion zones."""
        cursor = self.conn.execute(
            "SELECT id, name, polygon, created_at, updated_at FROM exclusion_zones"
        )
        zones = []
        for row in cursor.fetchall():
            zones.append({
                "id": row[0],
                "name": row[1],
                "polygon": json.loads(row[2]),
                "created_at": row[3],
                "updated_at": row[4],
            })
        return zones

    def update_exclusion_zone(self, zone_id: int, name: Optional[str] = None,
                              polygon: Optional[list[list[float]]] = None):
        """Update an exclusion zone's name and/or polygon."""
        now = datetime.now().isoformat()
        if name is not None and polygon is not None:
            self.conn.execute(
                "UPDATE exclusion_zones SET name = ?, polygon = ?, updated_at = ? WHERE id = ?",
                (name, json.dumps(polygon), now, zone_id)
            )
        elif name is not None:
            self.conn.execute(
                "UPDATE exclusion_zones SET name = ?, updated_at = ? WHERE id = ?",
                (name, now, zone_id)
            )
        elif polygon is not None:
            self.conn.execute(
                "UPDATE exclusion_zones SET polygon = ?, updated_at = ? WHERE id = ?",
                (json.dumps(polygon), now, zone_id)
            )
        self.conn.commit()

    def delete_exclusion_zone(self, zone_id: int):
        """Delete an exclusion zone."""
        self.conn.execute("DELETE FROM exclusion_zones WHERE id = ?", (zone_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()
