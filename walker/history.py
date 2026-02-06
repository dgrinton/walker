"""History database for tracking walked segments."""

import sqlite3
from datetime import datetime
from typing import Optional


class HistoryDB:
    """SQLite database for tracking walked segments"""

    def __init__(self, db_path: str = "walker_history.db"):
        self.conn = sqlite3.connect(db_path)
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

    def close(self):
        self.conn.close()
