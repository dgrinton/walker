#!/usr/bin/env python3
"""
Walker - Turn-by-turn pedestrian exploration guide
Prototype version

Usage:
    python walker.py [distance_km] [options]

Options:
    -v, --verbose     Audio updates every 30s, log to file every 10s
    --record FILE     Record GPS trace to JSON file for debugging
    --playback FILE   Playback GPS trace from JSON file
    --speed FACTOR    Playback speed multiplier (default: 1.0)
    --preview         Preview the calculated route without walking
    --execute         Execute route virtually and add to database (testing)
    --lat LAT         Starting latitude (for testing without GPS)
    --lon LON         Starting longitude (for testing without GPS)
    --html FILE       Output route visualization to HTML file (preview mode)
    --gpx FILE        Export route to GPX file (preview mode)
"""

import argparse
import json
import math
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import networkx as nx
import requests

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "gps_poll_interval": 3,  # seconds
    "intersection_arrival_radius": 15,  # meters
    "direction_warning_distance": 20,  # meters
    "route_deviation_threshold": 50,  # meters - recalculate if user deviates this far
    "default_walk_distance": 2000,  # meters (2km default)
    "osm_fetch_radius": 1500,  # meters - area to fetch from OSM
    "verbose_audio_interval": 30,  # seconds between audio status updates
    "verbose_log_interval": 10,  # seconds between log entries
    # Road type weights (lower = preferred)
    "road_weights": {
        "footway": 1,
        "pedestrian": 1,
        "path": 1,
        "residential": 2,
        "living_street": 2,
        "service": 3,
        "unclassified": 4,
        "tertiary": 5,
        "secondary": 7,
        "primary": 9,
        "trunk": 15,
        "motorway": 100,  # effectively blocked
    },
    "default_road_weight": 5,
}

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Location:
    lat: float
    lon: float
    accuracy: Optional[float] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Location":
        return cls(**d)


@dataclass
class Segment:
    """A street segment between two intersections"""
    id: str  # "{node1_id}-{node2_id}" sorted
    node1: int
    node2: int
    way_id: int
    name: Optional[str]
    road_type: str
    length: float  # meters

    @classmethod
    def make_id(cls, node1: int, node2: int) -> str:
        return f"{min(node1, node2)}-{max(node1, node2)}"


# =============================================================================
# Logging
# =============================================================================


class Logger:
    """Logs state to file"""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self.file = None
        if log_path:
            self.file = open(log_path, "a")
            self._write_header()

    def _write_header(self):
        if self.file:
            self.file.write(f"\n{'='*60}\n")
            self.file.write(f"Walker Log - {datetime.now().isoformat()}\n")
            self.file.write(f"{'='*60}\n\n")
            self.file.flush()

    def log(self, message: str, data: Optional[dict] = None):
        """Log a message with optional structured data"""
        timestamp = datetime.now().isoformat()
        line = f"[{timestamp}] {message}"
        if data:
            line += f" | {json.dumps(data)}"
        print(line)
        if self.file:
            self.file.write(line + "\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


# =============================================================================
# GPS Module
# =============================================================================


class GPS:
    """GPS access via Termux API"""

    def __init__(self):
        self.last_location: Optional[Location] = None
        self.consecutive_failures = 0

    def get_location(self, timeout: int = 30) -> Optional[Location]:
        """Get current location using termux-location"""
        try:
            result = subprocess.run(
                ["termux-location", "-p", "gps", "-r", "once"],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                self.consecutive_failures += 1
                error_msg = result.stderr.strip() if result.stderr else "unknown error"
                return None

            if not result.stdout or not result.stdout.strip():
                self.consecutive_failures += 1
                return None

            data = json.loads(result.stdout)
            location = Location(
                lat=data["latitude"],
                lon=data["longitude"],
                accuracy=data.get("accuracy"),
                timestamp=time.time()
            )
            self.last_location = location
            self.consecutive_failures = 0
            return location

        except subprocess.TimeoutExpired:
            self.consecutive_failures += 1
            return None
        except (json.JSONDecodeError, KeyError) as e:
            self.consecutive_failures += 1
            return None
        except FileNotFoundError:
            self.consecutive_failures += 1
            return None

    def get_status(self) -> str:
        """Get GPS status string"""
        if self.consecutive_failures == 0:
            acc = f", accuracy {self.last_location.accuracy:.0f}m" if self.last_location and self.last_location.accuracy else ""
            return f"GPS OK{acc}"
        else:
            return f"GPS: {self.consecutive_failures} consecutive failures"


class GPSRecorder:
    """Records GPS trace to file"""

    def __init__(self, gps: GPS, record_path: str):
        self.gps = gps
        self.record_path = record_path
        self.trace: list[dict] = []
        self.start_time = time.time()

    def get_location(self, timeout: int = 30) -> Optional[Location]:
        """Get location and record it"""
        location = self.gps.get_location(timeout)

        # Record even failed attempts
        entry = {
            "elapsed": time.time() - self.start_time,
            "timestamp": time.time(),
            "location": location.to_dict() if location else None,
            "status": self.gps.get_status()
        }
        self.trace.append(entry)

        return location

    def get_status(self) -> str:
        return self.gps.get_status()

    def save(self):
        """Save trace to file"""
        with open(self.record_path, "w") as f:
            json.dump({
                "recorded_at": datetime.now().isoformat(),
                "trace": self.trace
            }, f, indent=2)
        print(f"GPS trace saved to {self.record_path} ({len(self.trace)} entries)")


class GPSPlayback:
    """Plays back GPS trace from file"""

    def __init__(self, playback_path: str, speed: float = 1.0):
        self.playback_path = playback_path
        self.speed = speed
        self.trace: list[dict] = []
        self.index = 0
        self.last_location: Optional[Location] = None
        self.consecutive_failures = 0

        # Load trace
        with open(playback_path) as f:
            data = json.load(f)
            self.trace = data["trace"]
        print(f"Loaded GPS trace from {playback_path} ({len(self.trace)} entries)")

    def get_location(self, timeout: int = 30) -> Optional[Location]:
        """Get next location from trace sequentially"""
        if self.index >= len(self.trace):
            return None

        # Return entries one at a time (speed is handled by main loop sleep)
        entry = self.trace[self.index]
        self.index += 1

        if entry["location"]:
            location = Location.from_dict(entry["location"])
            self.last_location = location
            self.consecutive_failures = 0
            return location
        else:
            self.consecutive_failures += 1
            return None

    def get_poll_interval(self) -> float:
        """Get the interval to wait between polls based on trace timing and speed"""
        if self.index <= 0 or self.index >= len(self.trace):
            return CONFIG["gps_poll_interval"] / self.speed

        # Calculate time delta between current and previous entry
        prev_elapsed = self.trace[self.index - 1].get("elapsed", 0)
        curr_elapsed = self.trace[self.index].get("elapsed", 0)
        delta = curr_elapsed - prev_elapsed

        # Apply speed multiplier and clamp to reasonable range
        interval = delta / self.speed
        return max(0.1, min(interval, 5.0))

    def is_finished(self) -> bool:
        """Check if playback is complete"""
        return self.index >= len(self.trace)

    def get_status(self) -> str:
        progress = f"{self.index}/{len(self.trace)}"
        if self.consecutive_failures == 0:
            return f"Playback OK ({progress})"
        else:
            return f"Playback: {self.consecutive_failures} failures ({progress})"


# =============================================================================
# Math Utilities
# =============================================================================


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters"""
    R = 6371000  # Earth's radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing from point 1 to point 2 in degrees (0-360, 0=North)"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    x = math.sin(delta_lambda) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2) -
         math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda))

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def retry_with_backoff(func, max_time: float = 30.0, initial_delay: float = 1.0,
                       max_delay: float = 8.0, description: str = "operation"):
    """Retry a function with exponential backoff.

    Args:
        func: Function that returns a truthy value on success, falsy on failure
        max_time: Maximum total time to retry (seconds)
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        description: Description for logging

    Returns:
        The successful result, or None if all retries failed
    """
    start_time = time.time()
    delay = initial_delay
    attempt = 1

    while True:
        result = func()
        if result:
            return result

        elapsed = time.time() - start_time
        remaining = max_time - elapsed

        if remaining <= 0:
            print(f"{description} failed after {max_time:.0f}s")
            return None

        # Calculate next delay with exponential backoff
        actual_delay = min(delay, remaining, max_delay)
        print(f"{description} attempt {attempt} failed, retrying in {actual_delay:.1f}s...")
        time.sleep(actual_delay)

        delay = min(delay * 2, max_delay)
        attempt += 1


def bearing_to_compass(bearing: float) -> str:
    """Convert bearing to compass direction"""
    directions = ["north", "northeast", "east", "southeast",
                  "south", "southwest", "west", "northwest"]
    index = round(bearing / 45) % 8
    return directions[index]


def relative_direction(from_bearing: float, to_bearing: float) -> str:
    """Get relative direction (left, right, straight, etc.)"""
    diff = (to_bearing - from_bearing + 360) % 360

    if diff < 30 or diff > 330:
        return "straight"
    elif 30 <= diff < 60:
        return "slight right"
    elif 60 <= diff < 120:
        return "right"
    elif 120 <= diff < 150:
        return "sharp right"
    elif 150 <= diff < 210:
        return "u-turn"
    elif 210 <= diff < 240:
        return "sharp left"
    elif 240 <= diff < 300:
        return "left"
    else:
        return "slight left"


# =============================================================================
# OSM Module
# =============================================================================


class OSMFetcher:
    """Fetch street data from OpenStreetMap via Overpass API"""

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    @classmethod
    def fetch_streets(cls, lat: float, lon: float, radius: float) -> dict:
        """Fetch all walkable streets within radius of location"""

        # Overpass query for walkable ways
        query = f"""
        [out:json][timeout:30];
        (
          way["highway"~"^(footway|pedestrian|path|residential|living_street|service|unclassified|tertiary|secondary|primary)$"]
            (around:{radius},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """

        print(f"Fetching OSM data around ({lat:.5f}, {lon:.5f})...")

        try:
            response = requests.post(cls.OVERPASS_URL, data={"data": query}, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"OSM fetch error: {e}")
            return {"elements": []}


# =============================================================================
# Street Graph
# =============================================================================


class StreetGraph:
    """Graph representation of street network"""

    def __init__(self):
        self.graph = nx.Graph()
        self.nodes: dict[int, tuple[float, float]] = {}  # node_id -> (lat, lon)
        self.segments: dict[str, Segment] = {}  # segment_id -> Segment

    def build_from_osm(self, osm_data: dict):
        """Build graph from OSM data"""

        # First pass: collect all nodes
        for element in osm_data.get("elements", []):
            if element["type"] == "node":
                self.nodes[element["id"]] = (element["lat"], element["lon"])

        # Second pass: build edges from ways
        for element in osm_data.get("elements", []):
            if element["type"] != "way":
                continue

            tags = element.get("tags", {})
            road_type = tags.get("highway", "unclassified")
            name = tags.get("name")
            way_id = element["id"]
            nodes = element.get("nodes", [])

            # Create edges between consecutive nodes
            for i in range(len(nodes) - 1):
                n1, n2 = nodes[i], nodes[i + 1]

                if n1 not in self.nodes or n2 not in self.nodes:
                    continue

                lat1, lon1 = self.nodes[n1]
                lat2, lon2 = self.nodes[n2]
                length = haversine_distance(lat1, lon1, lat2, lon2)

                segment_id = Segment.make_id(n1, n2)
                segment = Segment(
                    id=segment_id,
                    node1=n1,
                    node2=n2,
                    way_id=way_id,
                    name=name,
                    road_type=road_type,
                    length=length
                )
                self.segments[segment_id] = segment

                # Add edge with weight based on road type
                weight = CONFIG["road_weights"].get(road_type, CONFIG["default_road_weight"])
                self.graph.add_edge(n1, n2,
                                    segment_id=segment_id,
                                    length=length,
                                    weight=weight * length,
                                    road_type=road_type,
                                    name=name)

        print(f"Built graph: {len(self.nodes)} nodes, {len(self.segments)} segments")

    def find_nearest_node(self, lat: float, lon: float) -> Optional[int]:
        """Find the nearest graph node to a location"""
        if not self.nodes:
            return None

        min_dist = float("inf")
        nearest = None

        for node_id, (nlat, nlon) in self.nodes.items():
            dist = haversine_distance(lat, lon, nlat, nlon)
            if dist < min_dist:
                min_dist = dist
                nearest = node_id

        return nearest

    def get_node_location(self, node_id: int) -> Optional[tuple[float, float]]:
        """Get lat/lon of a node"""
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: int) -> list[int]:
        """Get neighboring nodes"""
        return list(self.graph.neighbors(node_id))

    def get_segment(self, node1: int, node2: int) -> Optional[Segment]:
        """Get segment between two nodes"""
        segment_id = Segment.make_id(node1, node2)
        return self.segments.get(segment_id)

    def is_intersection(self, node_id: int) -> bool:
        """Check if a node is an intersection (degree > 2)"""
        return self.graph.degree(node_id) > 2


# =============================================================================
# History Database
# =============================================================================


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


# =============================================================================
# Audio Module
# =============================================================================


class Audio:
    """Text-to-speech for directions"""

    @staticmethod
    def speak(text: str):
        """Speak text using espeak (available in Termux)"""
        try:
            subprocess.run(
                ["espeak", "-s", "150", text],
                capture_output=True,
                timeout=10
            )
        except FileNotFoundError:
            # Fallback: try pyttsx3
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                print(f"[AUDIO] {text}")
        except Exception as e:
            print(f"Audio error: {e}")
            print(f"[AUDIO] {text}")


# =============================================================================
# Route Planner
# =============================================================================


class RoutePlanner:
    """Plans routes prioritizing novelty"""

    def __init__(self, graph: StreetGraph, history: HistoryDB):
        self.graph = graph
        self.history = history
        self.start_node: Optional[int] = None
        self.target_distance: float = 0
        self.walked_distance: float = 0
        self.current_path: list[int] = []
        self.planned_route: list[int] = []  # Full pre-calculated route
        self.route_index: int = 0  # Current position in planned_route

    def start_walk(self, start_node: int, target_distance: float):
        """Initialize a new walk"""
        self.start_node = start_node
        self.target_distance = target_distance
        self.walked_distance = 0
        self.current_path = [start_node]
        self.planned_route = []
        self.route_index = 0

    def calculate_full_route(self, start_node: int, target_distance: float) -> list[int]:
        """Calculate complete route from start, returning to start.

        Uses greedy algorithm prioritizing novelty while ensuring we can return.
        Returns list of node IDs representing the complete route.
        """
        self.start_node = start_node
        self.target_distance = target_distance

        route = [start_node]
        current = start_node
        previous = None
        distance = 0
        used_segments: set[str] = set()  # Track segments used in this route

        while True:
            # Get current location
            current_loc = self.graph.get_node_location(current)
            start_loc = self.graph.get_node_location(start_node)

            if not current_loc or not start_loc:
                break

            dist_to_start = haversine_distance(
                current_loc[0], current_loc[1], start_loc[0], start_loc[1]
            )
            remaining_budget = target_distance - distance

            # If we've walked enough and are close to start, head home
            if distance >= target_distance * 0.5 and remaining_budget < dist_to_start * 1.5:
                # Find path back to start
                try:
                    path_home = nx.shortest_path(
                        self.graph.graph, current, start_node, weight="length"
                    )
                    # Add remaining path (excluding current which is already in route)
                    for node in path_home[1:]:
                        segment = self.graph.get_segment(route[-1], node)
                        if segment:
                            distance += segment.length
                            used_segments.add(segment.id)
                        route.append(node)
                    break
                except nx.NetworkXNoPath:
                    break

            # Get neighbors
            neighbors = self.graph.get_neighbors(current)
            if not neighbors:
                break

            # Filter out where we came from (unless it's our only option)
            options = [n for n in neighbors if n != previous]
            if not options:
                options = neighbors

            # Filter options that would take us too far to return
            valid_options = []
            for n in options:
                n_loc = self.graph.get_node_location(n)
                if n_loc:
                    segment = self.graph.get_segment(current, n)
                    seg_len = segment.length if segment else 0
                    n_dist_to_start = haversine_distance(
                        n_loc[0], n_loc[1], start_loc[0], start_loc[1]
                    )
                    new_remaining = remaining_budget - seg_len
                    if new_remaining > n_dist_to_start * 0.9:
                        valid_options.append(n)

            if not valid_options:
                # No valid options, try to go home
                try:
                    path_home = nx.shortest_path(
                        self.graph.graph, current, start_node, weight="length"
                    )
                    for node in path_home[1:]:
                        segment = self.graph.get_segment(route[-1], node)
                        if segment:
                            distance += segment.length
                            used_segments.add(segment.id)
                        route.append(node)
                except nx.NetworkXNoPath:
                    pass
                break

            # Score all valid options and pick the best
            scored = [(n, self.score_edge(current, n, used_segments)) for n in valid_options]
            scored.sort(key=lambda x: x[1])

            # Pick best option
            next_node = scored[0][0]

            # Record segment
            segment = self.graph.get_segment(current, next_node)
            if segment:
                distance += segment.length
                used_segments.add(segment.id)

            route.append(next_node)
            previous = current
            current = next_node

            # Safety check for infinite loops
            if len(route) > 10000:
                break

        self.planned_route = route
        self.route_index = 0
        return route

    def get_route_distance(self) -> float:
        """Calculate total distance of planned route"""
        total = 0
        for i in range(len(self.planned_route) - 1):
            segment = self.graph.get_segment(self.planned_route[i], self.planned_route[i + 1])
            if segment:
                total += segment.length
        return total

    def get_route_segments(self) -> list[Segment]:
        """Get list of segments in planned route"""
        segments = []
        for i in range(len(self.planned_route) - 1):
            segment = self.graph.get_segment(self.planned_route[i], self.planned_route[i + 1])
            if segment:
                segments.append(segment)
        return segments

    def get_next_planned_node(self) -> Optional[int]:
        """Get the next node in the planned route"""
        if self.route_index + 1 < len(self.planned_route):
            return self.planned_route[self.route_index + 1]
        return None

    def advance_route(self):
        """Move to the next node in the planned route"""
        if self.route_index + 1 < len(self.planned_route):
            self.route_index += 1

    def get_current_planned_node(self) -> Optional[int]:
        """Get current node in planned route"""
        if 0 <= self.route_index < len(self.planned_route):
            return self.planned_route[self.route_index]
        return None

    def is_on_route(self, node_id: int) -> bool:
        """Check if a node is on the planned route ahead of current position"""
        remaining_route = self.planned_route[self.route_index:]
        return node_id in remaining_route

    def find_node_on_route(self, node_id: int) -> int:
        """Find index of node on route, or -1 if not found"""
        try:
            return self.planned_route.index(node_id, self.route_index)
        except ValueError:
            return -1

    def recalculate_from(self, current_node: int):
        """Recalculate route from current position to complete the walk"""
        remaining_distance = self.target_distance - self.walked_distance
        new_route = self.calculate_full_route(current_node, remaining_distance)
        # Update start node to original for return calculations
        self.start_node = self.planned_route[0] if self.planned_route else current_node
        return new_route

    def score_edge(self, from_node: int, to_node: int,
                   used_segments: Optional[set[str]] = None) -> float:
        """Score an edge - lower is better.

        Considers:
        - Novelty (never walked = 0, walked before = higher)
        - Road type preference
        - Whether it helps complete a loop
        - Segments already used in the current route calculation
        """
        segment = self.graph.get_segment(from_node, to_node)
        if not segment:
            return float("inf")

        # Base score from road type
        road_weight = CONFIG["road_weights"].get(
            segment.road_type, CONFIG["default_road_weight"]
        )

        # Novelty factor - heavily penalize previously walked segments
        times_walked, _ = self.history.get_segment_history(segment.id)

        # Also check if segment is already used in current route calculation
        used_in_route = used_segments and segment.id in used_segments

        if used_in_route:
            # Already used in this route - heavy penalty
            novelty_factor = 50 + (times_walked * 5)
        elif times_walked == 0:
            novelty_factor = 0  # Never walked - best!
        else:
            novelty_factor = 10 + (times_walked * 5)  # Penalty increases with visits

        # Distance to start (for loop completion)
        to_loc = self.graph.get_node_location(to_node)
        start_loc = self.graph.get_node_location(self.start_node)
        if to_loc and start_loc:
            dist_to_start = haversine_distance(
                to_loc[0], to_loc[1], start_loc[0], start_loc[1]
            )
            remaining_budget = self.target_distance - self.walked_distance

            # Penalize if this takes us too far to return
            if dist_to_start > remaining_budget * 0.9:
                return float("inf")  # Can't return in budget

        return road_weight + novelty_factor

    def choose_next_direction(self, current_node: int, came_from: Optional[int] = None) -> Optional[int]:
        """Choose the best next node to walk to"""
        neighbors = self.graph.get_neighbors(current_node)

        if not neighbors:
            return None

        # Filter out where we came from (unless it's our only option)
        options = [n for n in neighbors if n != came_from]
        if not options:
            options = neighbors

        # Check if we should head home
        remaining = self.target_distance - self.walked_distance
        current_loc = self.graph.get_node_location(current_node)
        start_loc = self.graph.get_node_location(self.start_node)

        if current_loc and start_loc:
            dist_to_start = haversine_distance(
                current_loc[0], current_loc[1], start_loc[0], start_loc[1]
            )

            # If we're close to distance budget, head home
            if remaining < dist_to_start * 1.2:
                # Find path back to start
                try:
                    path = nx.shortest_path(
                        self.graph.graph, current_node, self.start_node, weight="length"
                    )
                    if len(path) > 1:
                        return path[1]
                except nx.NetworkXNoPath:
                    pass

        # Score all options and pick the best
        scored = [(n, self.score_edge(current_node, n)) for n in options]
        scored.sort(key=lambda x: x[1])

        # Return best option (lowest score)
        return scored[0][0] if scored else None

    def get_direction_instruction(self, from_node: int, current_node: int, to_node: int) -> str:
        """Generate spoken direction instruction - always includes street name when known"""

        # Get bearings
        from_loc = self.graph.get_node_location(from_node)
        current_loc = self.graph.get_node_location(current_node)
        to_loc = self.graph.get_node_location(to_node)

        if not all([from_loc, current_loc, to_loc]):
            return "continue"

        # Calculate incoming and outgoing bearings
        incoming_bearing = bearing_between(
            from_loc[0], from_loc[1], current_loc[0], current_loc[1]
        )
        outgoing_bearing = bearing_between(
            current_loc[0], current_loc[1], to_loc[0], to_loc[1]
        )

        # Get relative direction
        rel_dir = relative_direction(incoming_bearing, outgoing_bearing)

        # Get segment info
        segment = self.graph.get_segment(current_node, to_node)

        # Always include street name when available
        if segment and segment.name:
            return f"{rel_dir} onto {segment.name}"
        else:
            # No street name - add compass heading for clarity
            compass = bearing_to_compass(outgoing_bearing)
            return f"{rel_dir}, heading {compass}"


# =============================================================================
# Main Walker App
# =============================================================================


class Walker:
    """Main application"""

    def __init__(self, verbose: bool = False, log_path: Optional[str] = None,
                 preview_mode: bool = False, execute_mode: bool = False,
                 start_location: Optional[tuple[float, float]] = None,
                 html_output: Optional[str] = None,
                 gpx_output: Optional[str] = None):
        self.gps = GPS()
        self.audio = Audio()
        self.history = HistoryDB()
        self.graph: Optional[StreetGraph] = None
        self.planner: Optional[RoutePlanner] = None
        self.verbose = verbose
        self.logger = Logger(log_path)
        self.preview_mode = preview_mode
        self.execute_mode = execute_mode
        self.start_location = start_location  # (lat, lon) tuple for testing
        self.html_output = html_output  # HTML file for route visualization
        self.gpx_output = gpx_output  # GPX file for navigation apps

        self.current_location: Optional[Location] = None
        self.current_node: Optional[int] = None
        self.previous_node: Optional[int] = None
        self.next_node: Optional[int] = None
        self.direction_given = False

        self.walk_id: Optional[int] = None
        self.segments_walked: int = 0

        # Verbose mode timing
        self.last_audio_update = 0
        self.last_log_update = 0
        self.walk_start_time = 0

        # GPS source (can be swapped for recording/playback)
        self.gps_source = self.gps

    def set_gps_source(self, source):
        """Set GPS source (GPS, GPSRecorder, or GPSPlayback)"""
        self.gps_source = source

    def get_state(self) -> dict:
        """Get current state as dict for logging"""
        state = {
            "walked_distance": self.planner.walked_distance if self.planner else 0,
            "target_distance": self.planner.target_distance if self.planner else 0,
            "segments_walked": self.segments_walked,
            "current_node": self.current_node,
            "next_node": self.next_node,
            "gps_status": self.gps_source.get_status() if hasattr(self.gps_source, 'get_status') else "unknown",
        }
        if self.current_location:
            state["location"] = {
                "lat": self.current_location.lat,
                "lon": self.current_location.lon,
                "accuracy": self.current_location.accuracy
            }
        return state

    def verbose_update(self):
        """Handle verbose mode updates"""
        now = time.time()

        # Log to file every 10 seconds
        if now - self.last_log_update >= CONFIG["verbose_log_interval"]:
            self.logger.log("STATE", self.get_state())
            self.last_log_update = now

        # Audio update every 30 seconds
        if now - self.last_audio_update >= CONFIG["verbose_audio_interval"]:
            elapsed = int(now - self.walk_start_time)
            minutes = elapsed // 60
            distance = int(self.planner.walked_distance) if self.planner else 0
            remaining = int(self.planner.target_distance - distance) if self.planner else 0

            status_parts = []
            if minutes > 0:
                status_parts.append(f"{minutes} minutes")
            status_parts.append(f"{distance} meters walked")
            status_parts.append(f"{remaining} to go")

            gps_status = self.gps_source.get_status() if hasattr(self.gps_source, 'get_status') else ""
            if "fail" in gps_status.lower():
                status_parts.append("GPS problems")

            status = ", ".join(status_parts)
            self.audio.speak(status)
            self.logger.log(f"AUDIO: {status}")
            self.last_audio_update = now

    def initialize(self, target_distance: float) -> bool:
        """Initialize walk with GPS fix and map data"""

        self.logger.log("Initializing walk", {"target_distance": target_distance})

        # Use provided start location if available (for testing)
        if self.start_location:
            lat, lon = self.start_location
            location = Location(lat=lat, lon=lon, accuracy=0, timestamp=time.time())
            self.logger.log("Using provided start location", {"lat": lat, "lon": lon})
            print(f"Using provided location: {lat:.5f}, {lon:.5f}")
        else:
            print("Getting GPS fix...")
            if not self.preview_mode and not self.execute_mode:
                self.audio.speak("Getting GPS fix")

            # Get GPS fix with retry and backoff (up to 30s)
            def try_gps():
                loc = self.gps_source.get_location(timeout=10)
                if loc:
                    self.logger.log("GPS fix obtained", {"lat": loc.lat, "lon": loc.lon})
                else:
                    self.logger.log("GPS attempt failed")
                return loc

            location = retry_with_backoff(
                try_gps,
                max_time=30.0,
                initial_delay=1.0,
                max_delay=8.0,
                description="GPS fix"
            )

            if not location:
                self.logger.log("Could not get GPS location after retries")
                print("Could not get GPS location")
                self.audio.speak("Could not get GPS location")
                return False

            self.logger.log("Got GPS fix", {"lat": location.lat, "lon": location.lon, "accuracy": location.accuracy})
            print(f"Location: {location.lat:.5f}, {location.lon:.5f} (accuracy: {location.accuracy}m)")

        self.current_location = location

        # Fetch OSM data with retry and backoff (up to 30s)
        osm_data = None

        def try_osm():
            data = OSMFetcher.fetch_streets(
                location.lat, location.lon, CONFIG["osm_fetch_radius"]
            )
            if data.get("elements"):
                self.logger.log("OSM data fetched", {"elements": len(data["elements"])})
                return data
            self.logger.log("OSM fetch returned no elements")
            return None

        osm_data = retry_with_backoff(
            try_osm,
            max_time=30.0,
            initial_delay=2.0,
            max_delay=8.0,
            description="Map data fetch"
        )

        if not osm_data:
            self.logger.log("Could not fetch map data after retries")
            print("Could not fetch map data")
            self.audio.speak("Could not fetch map data")
            return False

        # Build graph
        self.graph = StreetGraph()
        self.graph.build_from_osm(osm_data)
        self.logger.log("Built graph", {"nodes": len(self.graph.nodes), "segments": len(self.graph.segments)})

        # Find starting node
        self.current_node = self.graph.find_nearest_node(location.lat, location.lon)
        if not self.current_node:
            self.logger.log("Could not find starting point on map")
            print("Could not find starting point on map")
            return False

        node_loc = self.graph.get_node_location(self.current_node)
        self.logger.log("Starting node", {"node": self.current_node, "lat": node_loc[0], "lon": node_loc[1]})
        print(f"Starting at node {self.current_node} ({node_loc[0]:.5f}, {node_loc[1]:.5f})")

        # Initialize planner
        self.planner = RoutePlanner(self.graph, self.history)

        # Calculate the full route upfront
        print("Calculating route...")
        route = self.planner.calculate_full_route(self.current_node, target_distance)
        route_distance = self.planner.get_route_distance()
        self.logger.log("Route calculated", {
            "nodes": len(route),
            "distance": route_distance,
            "target": target_distance
        })
        print(f"Route calculated: {len(route)} nodes, {route_distance:.0f}m")

        # Preview mode: display route and optionally execute
        if self.preview_mode:
            self.display_route_preview()
            # If also executing, continue to execute_route
            if self.execute_mode:
                return True
            return False

        # Execute mode: run through route without GPS
        if self.execute_mode:
            return True

        # Start walk in database
        self.walk_id = self.history.start_walk()

        # Get first direction from planned route
        self.next_node = self.planner.get_next_planned_node()
        if self.next_node:
            next_loc = self.graph.get_node_location(self.next_node)
            bearing = bearing_between(
                node_loc[0], node_loc[1], next_loc[0], next_loc[1]
            )
            compass = bearing_to_compass(bearing)

            # Include street name if available
            segment = self.graph.get_segment(self.current_node, self.next_node)
            if segment and segment.name:
                direction = f"Head {compass} on {segment.name}"
            else:
                direction = f"Head {compass}"

            self.audio.speak(f"Walk started. {direction}")
            self.logger.log(f"Walk started, {direction}", {"next_node": self.next_node, "street": segment.name if segment else None})
            print(f"{direction} toward node {self.next_node}")

        self.walk_start_time = time.time()
        self.last_audio_update = time.time()
        self.last_log_update = time.time()

        return True

    def display_route_preview(self):
        """Display a preview of the calculated route"""
        if not self.planner or not self.planner.planned_route:
            print("No route to preview")
            return

        print("\n" + "=" * 60)
        print("ROUTE PREVIEW")
        print("=" * 60)

        segments = self.planner.get_route_segments()
        total_distance = self.planner.get_route_distance()

        print(f"\nTotal distance: {total_distance:.0f}m ({total_distance/1000:.2f}km)")
        print(f"Total segments: {len(segments)}")
        print(f"Total nodes: {len(self.planner.planned_route)}")

        # Count new vs walked segments
        new_segments = 0
        walked_segments = 0
        for seg in segments:
            times_walked, _ = self.history.get_segment_history(seg.id)
            if times_walked == 0:
                new_segments += 1
            else:
                walked_segments += 1

        print(f"\nNew segments: {new_segments}")
        print(f"Previously walked: {walked_segments}")
        if segments:
            print(f"Novelty: {new_segments / len(segments) * 100:.1f}%")

        print("\n" + "-" * 60)
        print("TURN-BY-TURN DIRECTIONS")
        print("-" * 60)

        route = self.planner.planned_route
        cumulative_distance = 0

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            segment = self.graph.get_segment(current_node, next_node)

            if not segment:
                continue

            # Check if this is an intersection (turn point)
            is_intersection = self.graph.is_intersection(next_node)

            # Get direction for this move
            current_loc = self.graph.get_node_location(current_node)
            next_loc = self.graph.get_node_location(next_node)

            if current_loc and next_loc:
                bearing = bearing_between(
                    current_loc[0], current_loc[1], next_loc[0], next_loc[1]
                )
                compass = bearing_to_compass(bearing)

                # Get relative direction if we have a previous node
                if i > 0:
                    prev_node = route[i - 1]
                    instruction = self.planner.get_direction_instruction(
                        prev_node, current_node, next_node
                    )
                else:
                    instruction = f"Head {compass}"

                # Only show at intersections or significant turns
                if i == 0 or is_intersection:
                    street_name = segment.name or "unnamed"
                    times_walked, _ = self.history.get_segment_history(segment.id)
                    novelty = " [NEW]" if times_walked == 0 else f" [x{times_walked}]"

                    print(f"\n{cumulative_distance:>6.0f}m | {instruction}")
                    print(f"         -> {street_name}{novelty} ({segment.length:.0f}m)")

            cumulative_distance += segment.length

        print(f"\n{cumulative_distance:>6.0f}m | Arrive at start")
        print("\n" + "=" * 60)

        # Generate HTML visualization if requested
        if self.html_output:
            self.generate_route_html()

        # Generate GPX file if requested
        if self.gpx_output:
            self.generate_route_gpx()

    def generate_route_html(self):
        """Generate an HTML file with a map visualization of the route"""
        if not self.planner or not self.planner.planned_route:
            print("No route to visualize")
            return

        route = self.planner.planned_route
        segments = self.planner.get_route_segments()

        # Collect coordinates for the route
        coordinates = []
        for node_id in route:
            loc = self.graph.get_node_location(node_id)
            if loc:
                coordinates.append(loc)

        if not coordinates:
            print("No coordinates to visualize")
            return

        # Calculate map center and bounds
        lats = [c[0] for c in coordinates]
        lons = [c[1] for c in coordinates]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        # Build segment data with colors based on novelty
        segment_data = []
        for i in range(len(route) - 1):
            node1, node2 = route[i], route[i + 1]
            loc1 = self.graph.get_node_location(node1)
            loc2 = self.graph.get_node_location(node2)
            segment = self.graph.get_segment(node1, node2)

            if loc1 and loc2 and segment:
                times_walked, _ = self.history.get_segment_history(segment.id)
                is_new = times_walked == 0
                street_name = segment.name or "unnamed"
                segment_data.append({
                    "coords": [[loc1[0], loc1[1]], [loc2[0], loc2[1]]],
                    "name": street_name,
                    "new": is_new,
                    "times_walked": times_walked,
                    "length": segment.length
                })

        # Build turn markers at intersections
        turn_markers = []
        for i, node_id in enumerate(route):
            if self.graph.is_intersection(node_id) or i == 0:
                loc = self.graph.get_node_location(node_id)
                if loc:
                    label = "Start" if i == 0 else f"Turn {len(turn_markers)}"
                    turn_markers.append({
                        "lat": loc[0],
                        "lon": loc[1],
                        "label": label
                    })

        # Generate HTML with Leaflet.js
        segments_json = json.dumps(segment_data)
        markers_json = json.dumps(turn_markers)
        total_distance = self.planner.get_route_distance()
        new_count = sum(1 for s in segment_data if s["new"])
        walked_count = len(segment_data) - new_count

        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Walker Route Preview</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        #map {{ position: absolute; top: 0; bottom: 0; left: 0; right: 300px; }}
        #sidebar {{ position: absolute; top: 0; bottom: 0; right: 0; width: 300px; background: #f5f5f5; overflow-y: auto; padding: 15px; box-sizing: border-box; }}
        h2 {{ margin-top: 0; }}
        .stat {{ margin: 10px 0; padding: 10px; background: white; border-radius: 5px; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .stat-value {{ font-size: 24px; font-weight: bold; }}
        .legend {{ margin-top: 20px; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .legend-color {{ width: 30px; height: 4px; margin-right: 10px; }}
        .new {{ background: #22c55e; }}
        .walked {{ background: #3b82f6; }}
        .segment-list {{ margin-top: 20px; max-height: 300px; overflow-y: auto; }}
        .segment-item {{ padding: 8px; background: white; margin: 5px 0; border-radius: 3px; font-size: 13px; border-left: 4px solid #ccc; }}
        .segment-item.new {{ border-left-color: #22c55e; }}
        .segment-item.walked {{ border-left-color: #3b82f6; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="sidebar">
        <h2>Route Preview</h2>
        <div class="stat">
            <div class="stat-label">Total Distance</div>
            <div class="stat-value">{total_distance/1000:.2f} km</div>
        </div>
        <div class="stat">
            <div class="stat-label">Segments</div>
            <div class="stat-value">{len(segment_data)}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Novelty</div>
            <div class="stat-value">{new_count / len(segment_data) * 100 if segment_data else 0:.0f}%</div>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-color new"></div> New segments ({new_count})</div>
            <div class="legend-item"><div class="legend-color walked"></div> Previously walked ({walked_count})</div>
        </div>
        <div class="segment-list">
            <h3>Segments</h3>
            <div id="segments"></div>
        </div>
    </div>
    <script>
        var segments = {segments_json};
        var markers = {markers_json};

        var map = L.map('map').setView([{center_lat}, {center_lon}], 15);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        // Draw route segments
        var bounds = [];
        segments.forEach(function(seg, idx) {{
            var color = seg.new ? '#22c55e' : '#3b82f6';
            var line = L.polyline(seg.coords, {{
                color: color,
                weight: 5,
                opacity: 0.8
            }}).addTo(map);
            line.bindPopup('<b>' + seg.name + '</b><br>' +
                          seg.length.toFixed(0) + 'm' +
                          (seg.new ? '<br><em>New!</em>' : '<br>Walked ' + seg.times_walked + 'x'));
            bounds.push(seg.coords[0]);
            bounds.push(seg.coords[1]);
        }});

        // Add turn markers
        markers.forEach(function(m, idx) {{
            var marker = L.circleMarker([m.lat, m.lon], {{
                radius: idx === 0 ? 10 : 6,
                fillColor: idx === 0 ? '#ef4444' : '#ffffff',
                color: '#333',
                weight: 2,
                fillOpacity: 1
            }}).addTo(map);
            marker.bindPopup(m.label);
        }});

        // Fit map to route bounds
        if (bounds.length > 0) {{
            map.fitBounds(bounds, {{ padding: [20, 20] }});
        }}

        // Build segment list
        var segList = document.getElementById('segments');
        segments.forEach(function(seg, idx) {{
            var div = document.createElement('div');
            div.className = 'segment-item ' + (seg.new ? 'new' : 'walked');
            div.innerHTML = '<strong>' + seg.name + '</strong><br>' +
                           seg.length.toFixed(0) + 'm' +
                           (seg.new ? ' - New' : ' - Walked ' + seg.times_walked + 'x');
            segList.appendChild(div);
        }});
    </script>
</body>
</html>'''

        # Write HTML file
        with open(self.html_output, 'w') as f:
            f.write(html)

        print(f"\nRoute visualization saved to: {self.html_output}")

    def generate_route_gpx(self):
        """Generate a GPX file for use in OsmAnd or other GPS navigation apps"""
        if not self.planner or not self.planner.planned_route:
            print("No route to export")
            return

        route = self.planner.planned_route
        total_distance = self.planner.get_route_distance()
        timestamp = datetime.now().isoformat()

        # Build waypoints at intersections with turn instructions
        waypoints = []
        for i, node_id in enumerate(route):
            loc = self.graph.get_node_location(node_id)
            if not loc:
                continue

            # Only add waypoints at start, intersections, and end
            is_start = (i == 0)
            is_end = (i == len(route) - 1)
            is_intersection = self.graph.is_intersection(node_id)

            if is_start or is_end or is_intersection:
                if is_start:
                    name = "Start"
                elif is_end:
                    name = "End"
                else:
                    # Get turn instruction for this intersection
                    if i > 0 and i < len(route) - 1:
                        prev_node = route[i - 1]
                        next_node = route[i + 1]
                        instruction = self.planner.get_direction_instruction(
                            prev_node, node_id, next_node
                        )
                        name = instruction.capitalize()
                    else:
                        name = f"Waypoint {len(waypoints) + 1}"

                waypoints.append({
                    "lat": loc[0],
                    "lon": loc[1],
                    "name": name
                })

        # Build track points for the route line
        trackpoints = []
        for node_id in route:
            loc = self.graph.get_node_location(node_id)
            if loc:
                trackpoints.append({"lat": loc[0], "lon": loc[1]})

        # Generate GPX XML
        gpx_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="Walker"',
            '     xmlns="http://www.topografix.com/GPX/1/1"',
            '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '     xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">',
            '  <metadata>',
            f'    <name>Walker Route ({total_distance/1000:.2f} km)</name>',
            f'    <time>{timestamp}</time>',
            '  </metadata>',
        ]

        # Add waypoints
        for wp in waypoints:
            gpx_lines.append(f'  <wpt lat="{wp["lat"]:.6f}" lon="{wp["lon"]:.6f}">')
            gpx_lines.append(f'    <name>{wp["name"]}</name>')
            gpx_lines.append('  </wpt>')

        # Add track
        gpx_lines.append('  <trk>')
        gpx_lines.append(f'    <name>Walker Route</name>')
        gpx_lines.append('    <trkseg>')
        for tp in trackpoints:
            gpx_lines.append(f'      <trkpt lat="{tp["lat"]:.6f}" lon="{tp["lon"]:.6f}"/>')
        gpx_lines.append('    </trkseg>')
        gpx_lines.append('  </trk>')
        gpx_lines.append('</gpx>')

        # Write GPX file
        gpx_content = '\n'.join(gpx_lines)
        with open(self.gpx_output, 'w') as f:
            f.write(gpx_content)

        print(f"\nGPX route saved to: {self.gpx_output}")
        print(f"  {len(waypoints)} waypoints, {len(trackpoints)} track points")
        print("  Import into OsmAnd: Menu -> My Places -> Tracks -> Import")

    def execute_route(self):
        """Execute the planned route virtually and add to database"""
        if not self.planner or not self.planner.planned_route:
            print("No route to execute")
            return

        print("\n" + "=" * 60)
        print("EXECUTING ROUTE (Virtual Walk)")
        print("=" * 60)

        # Start walk in database
        self.walk_id = self.history.start_walk()

        route = self.planner.planned_route
        total_distance = 0

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            segment = self.graph.get_segment(current_node, next_node)

            if segment:
                # Record this segment as walked
                self.history.record_segment(segment.id)
                total_distance += segment.length
                self.segments_walked += 1

                street_name = segment.name or "unnamed"
                print(f"  Walked: {street_name} ({segment.length:.0f}m) - Total: {total_distance:.0f}m")

        # End walk in database
        self.history.end_walk(self.walk_id, total_distance, self.segments_walked)

        print("\n" + "-" * 60)
        print("EXECUTION COMPLETE")
        print(f"  Total distance: {total_distance:.0f}m ({total_distance/1000:.2f}km)")
        print(f"  Segments walked: {self.segments_walked}")
        print("  All segments recorded to database")
        print("=" * 60)

    def update(self) -> bool:
        """Main update loop - returns False when walk is complete"""

        # Verbose mode updates
        if self.verbose:
            self.verbose_update()

        # Get current GPS
        location = self.gps_source.get_location()
        if not location:
            self.logger.log("GPS fix failed", {"status": self.gps_source.get_status() if hasattr(self.gps_source, 'get_status') else "unknown"})
            return True  # Keep going even with GPS errors

        self.current_location = location

        # Find nearest node to current GPS position
        nearest = self.graph.find_nearest_node(location.lat, location.lon)
        if not nearest:
            return True

        nearest_loc = self.graph.get_node_location(nearest)
        dist_to_nearest = haversine_distance(
            location.lat, location.lon, nearest_loc[0], nearest_loc[1]
        )

        # Calculate distance to next planned node
        dist_to_next = float("inf")
        if self.next_node:
            next_loc = self.graph.get_node_location(self.next_node)
            dist_to_next = haversine_distance(
                location.lat, location.lon, next_loc[0], next_loc[1]
            )

            # Approaching next node - give direction warning
            if (dist_to_next < CONFIG["direction_warning_distance"] and
                not self.direction_given and
                self.graph.is_intersection(self.next_node)):

                # Look ahead in planned route for next direction
                route_idx = self.planner.find_node_on_route(self.next_node)
                if route_idx >= 0 and route_idx + 1 < len(self.planner.planned_route):
                    after_next = self.planner.planned_route[route_idx + 1]
                    instruction = self.planner.get_direction_instruction(
                        self.current_node, self.next_node, after_next
                    )
                    self.audio.speak(instruction)
                    self.logger.log(f"Direction: {instruction}", {"at_node": self.next_node, "next": after_next})
                    print(f"Direction: {instruction}")
                    self.direction_given = True

        # Check for route deviation
        deviation_detected = self._check_route_deviation(location, nearest, dist_to_nearest)

        # Determine what happened:
        # 1. Arrived at expected next_node (on planned route)
        # 2. Deviated too far from route (needs recalculation)
        # 3. Still at current node (no movement)

        arrived_at_next = self.next_node and dist_to_next < CONFIG["intersection_arrival_radius"]

        if arrived_at_next:
            # Arrived at expected next node on planned route
            segment = self.graph.get_segment(self.current_node, self.next_node)
            if segment:
                self.history.record_segment(segment.id)
                self.planner.walked_distance += segment.length
                self.segments_walked += 1
                self.logger.log(f"Walked segment", {
                    "segment": segment.id,
                    "length": segment.length,
                    "total": self.planner.walked_distance,
                    "name": segment.name
                })
                print(f"Walked {segment.length:.0f}m, total: {self.planner.walked_distance:.0f}m")

            # Check if walk is complete (arrived back at start)
            if (self.next_node == self.planner.start_node and
                self.planner.walked_distance >= self.planner.target_distance * 0.8):
                self.audio.speak("Walk complete")
                self.logger.log("Walk complete")
                print("Walk complete!")
                return False

            # Advance along planned route
            self.previous_node = self.current_node
            self.current_node = self.next_node
            self.planner.advance_route()
            self.next_node = self.planner.get_next_planned_node()
            self.direction_given = False

            if not self.next_node:
                # Reached end of planned route
                if self.current_node == self.planner.start_node:
                    self.audio.speak("Walk complete")
                    self.logger.log("Walk complete")
                    print("Walk complete!")
                else:
                    self.logger.log("End of planned route")
                    print("End of planned route")
                return False

        elif deviation_detected:
            # User deviated too far from planned route - recalculate
            self.logger.log("Route deviation detected", {
                "from": self.current_node,
                "to": nearest,
                "expected": self.next_node,
                "dist_to_nearest": dist_to_nearest
            })
            print(f"Route deviation detected - recalculating from node {nearest}")

            # Record the segment they actually walked if it exists
            segment = self.graph.get_segment(self.current_node, nearest)
            if segment:
                self.history.record_segment(segment.id)
                self.planner.walked_distance += segment.length
                self.segments_walked += 1
                self.logger.log(f"Walked segment (deviation)", {
                    "segment": segment.id,
                    "length": segment.length,
                    "total": self.planner.walked_distance
                })
                print(f"Walked {segment.length:.0f}m (deviation), total: {self.planner.walked_distance:.0f}m")

            # Recalculate route from current position
            self.previous_node = self.current_node
            self.current_node = nearest
            self._recalculate_route()

        return True

    def _check_route_deviation(self, location: Location, nearest_node: int,
                                dist_to_nearest: float) -> bool:
        """Check if user has deviated too far from the planned route.

        Returns True if recalculation is needed.
        """
        if not self.planner.planned_route:
            return False

        # Check if nearest node is on the planned route ahead
        if self.planner.is_on_route(nearest_node):
            return False

        # Check distance from GPS position to the expected next node
        if self.next_node:
            next_loc = self.graph.get_node_location(self.next_node)
            if next_loc:
                dist_to_next = haversine_distance(
                    location.lat, location.lon, next_loc[0], next_loc[1]
                )
                # If we're far from both the expected path and any known node
                if dist_to_next > CONFIG["route_deviation_threshold"]:
                    # Confirm we've actually moved to a different node
                    if nearest_node != self.current_node and dist_to_nearest < CONFIG["intersection_arrival_radius"]:
                        return True

        return False

    def _recalculate_route(self):
        """Recalculate route from current position"""
        remaining_distance = self.planner.target_distance - self.planner.walked_distance
        print(f"Recalculating route for remaining {remaining_distance:.0f}m...")

        # Store original start for return
        original_start = self.planner.start_node

        # Calculate new route
        new_route = self.planner.calculate_full_route(self.current_node, remaining_distance)

        # Restore original start for proper loop completion
        self.planner.start_node = original_start

        self.logger.log("Route recalculated", {
            "new_nodes": len(new_route),
            "remaining_distance": remaining_distance
        })
        print(f"New route: {len(new_route)} nodes")

        # Get next direction
        self.next_node = self.planner.get_next_planned_node()
        self.direction_given = False

        if self.next_node:
            # Give immediate direction guidance
            route_idx = self.planner.find_node_on_route(self.next_node)
            if route_idx >= 0 and route_idx + 1 < len(self.planner.planned_route):
                after_next = self.planner.planned_route[route_idx + 1]
                if self.graph.is_intersection(self.next_node):
                    instruction = self.planner.get_direction_instruction(
                        self.current_node, self.next_node, after_next
                    )
                    self.audio.speak(f"Recalculating. {instruction}")
                    self.logger.log(f"Recalculated: {instruction}")
                    print(f"Recalculated: {instruction}")

    def get_poll_interval(self) -> float:
        """Get poll interval, respecting playback speed if applicable"""
        if isinstance(self.gps_source, GPSPlayback):
            return self.gps_source.get_poll_interval()
        return CONFIG["gps_poll_interval"]

    def is_playback_finished(self) -> bool:
        """Check if playback is complete"""
        if isinstance(self.gps_source, GPSPlayback):
            return self.gps_source.is_finished()
        return False

    def run(self, target_distance: float):
        """Run the walk"""

        print(f"\n=== Walker ===")
        print(f"Target distance: {target_distance}m")
        if self.preview_mode:
            print("Mode: PREVIEW (calculate and display route)")
        elif self.execute_mode:
            print("Mode: EXECUTE (virtual walk for testing)")
        else:
            if self.verbose:
                print("Verbose mode: ON (audio every 30s, log every 10s)")
            if isinstance(self.gps_source, GPSPlayback):
                print(f"Playback mode: {self.gps_source.speed}x speed")
            print("Press Ctrl+C to stop")
        print()

        if not self.initialize(target_distance):
            # Preview mode exits after displaying route
            if self.preview_mode:
                self.history.close()
                self.logger.close()
            return

        # Execute mode: run through route virtually
        if self.execute_mode:
            self.execute_route()
            self.history.close()
            self.logger.close()
            return

        # Normal walking mode
        try:
            while self.update():
                # Check if playback finished
                if self.is_playback_finished():
                    print("\nPlayback finished")
                    self.logger.log("Playback finished")
                    break
                time.sleep(self.get_poll_interval())
        except KeyboardInterrupt:
            print("\nWalk interrupted")
            self.audio.speak("Walk ended")
            self.logger.log("Walk interrupted by user")
        finally:
            # Record walk stats
            if self.walk_id:
                self.history.end_walk(
                    self.walk_id,
                    self.planner.walked_distance if self.planner else 0,
                    self.segments_walked
                )

            # Save GPS recording if applicable
            if isinstance(self.gps_source, GPSRecorder):
                self.gps_source.save()

            summary = {
                "distance": self.planner.walked_distance if self.planner else 0,
                "segments": self.segments_walked,
                "duration": time.time() - self.walk_start_time if self.walk_start_time else 0
            }
            self.logger.log("Walk summary", summary)

            print(f"\nWalk summary:")
            print(f"  Distance: {summary['distance']:.0f}m")
            print(f"  Segments: {summary['segments']}")
            print(f"  Duration: {summary['duration']/60:.1f} minutes")

            self.history.close()
            self.logger.close()


# =============================================================================
# Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Walker - Turn-by-turn pedestrian exploration guide"
    )
    parser.add_argument("distance", type=float, nargs="?", default=2.0,
                        help="Target distance in km (default: 2.0)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Audio updates every 30s, log to file every 10s")
    parser.add_argument("--record", metavar="FILE",
                        help="Record GPS trace to JSON file")
    parser.add_argument("--playback", metavar="FILE",
                        help="Playback GPS trace from JSON file")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--log", metavar="FILE",
                        help="Log file path (default: walker_TIMESTAMP.log in verbose mode)")
    parser.add_argument("--preview", action="store_true",
                        help="Preview the calculated route without walking")
    parser.add_argument("--execute", action="store_true",
                        help="Execute route virtually and add to database (testing)")
    parser.add_argument("--lat", type=float, metavar="LAT",
                        help="Starting latitude (for testing without GPS)")
    parser.add_argument("--lon", type=float, metavar="LON",
                        help="Starting longitude (for testing without GPS)")
    parser.add_argument("--html", metavar="FILE",
                        help="Output route visualization to HTML file (preview mode)")
    parser.add_argument("--gpx", metavar="FILE",
                        help="Export route to GPX file (preview mode)")

    args = parser.parse_args()

    # Validate lat/lon - must provide both or neither
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be used together")

    # Convert km to meters
    distance = args.distance * 1000

    # Determine log path
    log_path = args.log
    if args.verbose and not log_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"walker_{timestamp}.log"

    # Create walker
    start_location = (args.lat, args.lon) if args.lat is not None else None
    walker = Walker(
        verbose=args.verbose,
        log_path=log_path,
        preview_mode=args.preview,
        execute_mode=args.execute,
        start_location=start_location,
        html_output=args.html,
        gpx_output=args.gpx
    )

    # Set up GPS source
    if args.playback:
        if not Path(args.playback).exists():
            print(f"Playback file not found: {args.playback}")
            sys.exit(1)
        walker.set_gps_source(GPSPlayback(args.playback, args.speed))
    elif args.record:
        walker.set_gps_source(GPSRecorder(walker.gps, args.record))

    walker.run(distance)


if __name__ == "__main__":
    main()
