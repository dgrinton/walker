#!/usr/bin/env python3
"""
Walker - Turn-by-turn pedestrian exploration guide
Prototype version
"""

import json
import math
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
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
    "default_walk_distance": 2000,  # meters (2km default)
    "osm_fetch_radius": 1500,  # meters - area to fetch from OSM
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
# GPS Module
# =============================================================================


class GPS:
    """GPS access via Termux API"""

    @staticmethod
    def get_location(timeout: int = 30) -> Optional[Location]:
        """Get current location using termux-location"""
        try:
            result = subprocess.run(
                ["termux-location", "-p", "gps", "-r", "once"],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode != 0:
                print(f"GPS error: {result.stderr}")
                return None

            data = json.loads(result.stdout)
            return Location(
                lat=data["latitude"],
                lon=data["longitude"],
                accuracy=data.get("accuracy"),
                timestamp=data.get("elapsedMs", time.time() * 1000) / 1000
            )
        except subprocess.TimeoutExpired:
            print("GPS timeout")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            print(f"GPS parse error: {e}")
            return None
        except FileNotFoundError:
            print("termux-location not found. Install with: pkg install termux-api")
            return None


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

    def start_walk(self, start_node: int, target_distance: float):
        """Initialize a new walk"""
        self.start_node = start_node
        self.target_distance = target_distance
        self.walked_distance = 0
        self.current_path = [start_node]

    def score_edge(self, from_node: int, to_node: int) -> float:
        """Score an edge - lower is better.

        Considers:
        - Novelty (never walked = 0, walked before = higher)
        - Road type preference
        - Whether it helps complete a loop
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
        if times_walked == 0:
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
        """Generate spoken direction instruction"""

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

        # For simple intersections, just say the direction
        num_options = len(self.graph.get_neighbors(current_node))

        if num_options <= 3 and rel_dir in ["left", "right", "straight"]:
            return rel_dir

        # For complex intersections, add street name or compass heading
        if segment and segment.name:
            return f"{rel_dir} onto {segment.name}"
        else:
            compass = bearing_to_compass(outgoing_bearing)
            return f"{rel_dir}, heading {compass}"


# =============================================================================
# Main Walker App
# =============================================================================


class Walker:
    """Main application"""

    def __init__(self):
        self.gps = GPS()
        self.audio = Audio()
        self.history = HistoryDB()
        self.graph: Optional[StreetGraph] = None
        self.planner: Optional[RoutePlanner] = None

        self.current_location: Optional[Location] = None
        self.current_node: Optional[int] = None
        self.previous_node: Optional[int] = None
        self.next_node: Optional[int] = None
        self.direction_given = False

        self.walk_id: Optional[int] = None
        self.segments_walked: int = 0

    def initialize(self, target_distance: float):
        """Initialize walk with GPS fix and map data"""

        print("Getting GPS fix...")
        self.audio.speak("Getting GPS fix")

        location = self.gps.get_location()
        if not location:
            print("Could not get GPS location")
            return False

        self.current_location = location
        print(f"Location: {location.lat:.5f}, {location.lon:.5f}")

        # Fetch OSM data
        osm_data = OSMFetcher.fetch_streets(
            location.lat, location.lon, CONFIG["osm_fetch_radius"]
        )

        if not osm_data.get("elements"):
            print("Could not fetch map data")
            return False

        # Build graph
        self.graph = StreetGraph()
        self.graph.build_from_osm(osm_data)

        # Find starting node
        self.current_node = self.graph.find_nearest_node(location.lat, location.lon)
        if not self.current_node:
            print("Could not find starting point on map")
            return False

        node_loc = self.graph.get_node_location(self.current_node)
        print(f"Starting at node {self.current_node} ({node_loc[0]:.5f}, {node_loc[1]:.5f})")

        # Initialize planner
        self.planner = RoutePlanner(self.graph, self.history)
        self.planner.start_walk(self.current_node, target_distance)

        # Start walk in database
        self.walk_id = self.history.start_walk()

        # Choose first direction
        self.next_node = self.planner.choose_next_direction(self.current_node)
        if self.next_node:
            next_loc = self.graph.get_node_location(self.next_node)
            bearing = bearing_between(
                node_loc[0], node_loc[1], next_loc[0], next_loc[1]
            )
            compass = bearing_to_compass(bearing)

            self.audio.speak(f"Walk started. Head {compass}")
            print(f"Head {compass} toward node {self.next_node}")

        return True

    def update(self) -> bool:
        """Main update loop - returns False when walk is complete"""

        # Get current GPS
        location = self.gps.get_location()
        if not location:
            return True  # Keep going even with GPS errors

        self.current_location = location

        # Find nearest node
        nearest = self.graph.find_nearest_node(location.lat, location.lon)
        if not nearest:
            return True

        nearest_loc = self.graph.get_node_location(nearest)
        dist_to_nearest = haversine_distance(
            location.lat, location.lon, nearest_loc[0], nearest_loc[1]
        )

        # Check if we've reached the next node
        if self.next_node:
            next_loc = self.graph.get_node_location(self.next_node)
            dist_to_next = haversine_distance(
                location.lat, location.lon, next_loc[0], next_loc[1]
            )

            # Approaching - give direction warning
            if (dist_to_next < CONFIG["direction_warning_distance"] and
                not self.direction_given and
                self.graph.is_intersection(self.next_node)):

                # Calculate and announce next direction
                after_next = self.planner.choose_next_direction(
                    self.next_node, self.current_node
                )
                if after_next:
                    instruction = self.planner.get_direction_instruction(
                        self.current_node, self.next_node, after_next
                    )
                    self.audio.speak(instruction)
                    print(f"Direction: {instruction}")
                    self.direction_given = True

            # Arrived at next node
            if dist_to_next < CONFIG["intersection_arrival_radius"]:
                # Record segment walked
                segment = self.graph.get_segment(self.current_node, self.next_node)
                if segment:
                    self.history.record_segment(segment.id)
                    self.planner.walked_distance += segment.length
                    self.segments_walked += 1
                    print(f"Walked {segment.length:.0f}m, total: {self.planner.walked_distance:.0f}m")

                # Check if walk is complete
                if (self.next_node == self.planner.start_node and
                    self.planner.walked_distance >= self.planner.target_distance * 0.8):
                    self.audio.speak("Walk complete")
                    print("Walk complete!")
                    return False

                # Move to next node
                self.previous_node = self.current_node
                self.current_node = self.next_node
                self.next_node = self.planner.choose_next_direction(
                    self.current_node, self.previous_node
                )
                self.direction_given = False

                if not self.next_node:
                    print("No path available")
                    self.audio.speak("No path available")
                    return False

        # Check if user deviated
        elif nearest != self.current_node:
            # User went somewhere unexpected - adapt
            print(f"Deviation detected: expected {self.current_node}, now at {nearest}")

            # Record the segment they actually walked if it exists
            segment = self.graph.get_segment(self.current_node, nearest)
            if segment:
                self.history.record_segment(segment.id)
                self.planner.walked_distance += segment.length
                self.segments_walked += 1

            # Update position and recalculate
            self.previous_node = self.current_node
            self.current_node = nearest
            self.next_node = self.planner.choose_next_direction(
                self.current_node, self.previous_node
            )
            self.direction_given = False

            if self.next_node:
                # Give new direction
                after_next = self.planner.choose_next_direction(
                    self.next_node, self.current_node
                )
                if after_next and self.graph.is_intersection(self.next_node):
                    instruction = self.planner.get_direction_instruction(
                        self.current_node, self.next_node, after_next
                    )
                    self.audio.speak(f"Recalculating. {instruction}")
                    print(f"Recalculated: {instruction}")

        return True

    def run(self, target_distance: float):
        """Run the walk"""

        print(f"\n=== Walker Prototype ===")
        print(f"Target distance: {target_distance}m")
        print("Press Ctrl+C to stop\n")

        if not self.initialize(target_distance):
            return

        try:
            while self.update():
                time.sleep(CONFIG["gps_poll_interval"])
        except KeyboardInterrupt:
            print("\nWalk interrupted")
            self.audio.speak("Walk ended")
        finally:
            # Record walk stats
            if self.walk_id:
                self.history.end_walk(
                    self.walk_id,
                    self.planner.walked_distance if self.planner else 0,
                    self.segments_walked
                )

            print(f"\nWalk summary:")
            print(f"  Distance: {self.planner.walked_distance:.0f}m" if self.planner else "  Distance: unknown")
            print(f"  Segments: {self.segments_walked}")

            self.history.close()


# =============================================================================
# Entry Point
# =============================================================================


def main():
    import sys

    # Get target distance from command line or use default
    if len(sys.argv) > 1:
        try:
            distance = float(sys.argv[1]) * 1000  # Convert km to meters
        except ValueError:
            print(f"Usage: {sys.argv[0]} [distance_km]")
            print(f"Example: {sys.argv[0]} 2.5  (for 2.5km walk)")
            sys.exit(1)
    else:
        distance = CONFIG["default_walk_distance"]

    walker = Walker()
    walker.run(distance)


if __name__ == "__main__":
    main()
