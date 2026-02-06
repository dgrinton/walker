"""Street graph representation."""

from typing import Optional

import networkx as nx

from .config import CONFIG
from .models import Segment
from .geo import haversine_distance, bearing_between, are_bearings_parallel


class StreetGraph:
    """Graph representation of street network"""

    def __init__(self):
        self.graph = nx.Graph()
        self.nodes: dict[int, tuple[float, float]] = {}  # node_id -> (lat, lon)
        self.segments: dict[str, Segment] = {}  # segment_id -> Segment
        self.parallel_segments: dict[str, set[str]] = {}  # segment_id -> set of parallel segment_ids

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

        self._flag_busy_road_adjacent()
        self._compute_parallel_segments()
        print(f"Built graph: {len(self.nodes)} nodes, {len(self.segments)} segments")

    def _flag_busy_road_adjacent(self):
        """Flag footpath segments that run alongside busy roads."""
        busy_road_types = CONFIG["busy_road_types"]
        footpath_types = CONFIG["footpath_types"]
        threshold = CONFIG["busy_road_proximity_threshold"]

        # Collect midpoints of all busy road segments
        busy_midpoints = []
        for segment in self.segments.values():
            if segment.road_type in busy_road_types:
                loc1 = self.nodes.get(segment.node1)
                loc2 = self.nodes.get(segment.node2)
                if loc1 and loc2:
                    mid = ((loc1[0] + loc2[0]) / 2, (loc1[1] + loc2[1]) / 2)
                    busy_midpoints.append(mid)

        if not busy_midpoints:
            return

        # Check each footpath segment against busy road midpoints
        flagged = 0
        for segment in self.segments.values():
            if segment.road_type not in footpath_types:
                continue
            loc1 = self.nodes.get(segment.node1)
            loc2 = self.nodes.get(segment.node2)
            if not loc1 or not loc2:
                continue
            mid = ((loc1[0] + loc2[0]) / 2, (loc1[1] + loc2[1]) / 2)
            for busy_mid in busy_midpoints:
                if haversine_distance(mid[0], mid[1], busy_mid[0], busy_mid[1]) < threshold:
                    segment.busy_road_adjacent = True
                    flagged += 1
                    break

        if flagged:
            print(f"Flagged {flagged} footpath segments adjacent to busy roads")

    def _compute_parallel_segments(self):
        """Precompute mapping of parallel segments for corridor deduplication.

        Two segments are parallel if they belong to different OSM ways, share no
        nodes, have midpoints within the proximity threshold, and bearings within
        the angle threshold (direction-agnostic).
        """
        proximity = CONFIG["parallel_distance_threshold"]
        angle_threshold = CONFIG["parallel_angle_threshold"]
        min_length = CONFIG["corridor_min_segment_length"]

        # Pre-compute midpoints and bearings for eligible segments
        seg_info: list[tuple[str, Segment, float, float, float]] = []  # (id, seg, mid_lat, mid_lon, bearing)
        for seg in self.segments.values():
            if seg.length < min_length:
                continue
            loc1 = self.nodes.get(seg.node1)
            loc2 = self.nodes.get(seg.node2)
            if not loc1 or not loc2:
                continue
            mid_lat = (loc1[0] + loc2[0]) / 2
            mid_lon = (loc1[1] + loc2[1]) / 2
            seg_bearing = bearing_between(loc1[0], loc1[1], loc2[0], loc2[1])
            seg_info.append((seg.id, seg, mid_lat, mid_lon, seg_bearing))

        # Sort by latitude for pre-filtering
        seg_info.sort(key=lambda x: x[2])

        # Approximate latitude degrees per meter (for pre-filter)
        lat_per_meter = 1.0 / 111_320
        lat_margin = proximity * lat_per_meter

        pair_count = 0
        n = len(seg_info)
        for i in range(n):
            id_a, seg_a, lat_a, lon_a, bearing_a = seg_info[i]
            nodes_a = {seg_a.node1, seg_a.node2}
            for j in range(i + 1, n):
                id_b, seg_b, lat_b, lon_b, bearing_b = seg_info[j]
                # Latitude pre-filter: skip if too far apart
                if lat_b - lat_a > lat_margin:
                    break
                # Must be from different OSM ways
                if seg_a.way_id == seg_b.way_id:
                    continue
                # Must not share nodes
                if seg_b.node1 in nodes_a or seg_b.node2 in nodes_a:
                    continue
                # Check bearing parallelism
                if not are_bearings_parallel(bearing_a, bearing_b, angle_threshold):
                    continue
                # Check midpoint proximity
                if haversine_distance(lat_a, lon_a, lat_b, lon_b) >= proximity:
                    continue
                # Record bidirectional mapping
                self.parallel_segments.setdefault(id_a, set()).add(id_b)
                self.parallel_segments.setdefault(id_b, set()).add(id_a)
                pair_count += 1

        if pair_count:
            print(f"Detected {pair_count} parallel segment pairs for corridor deduplication")

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
