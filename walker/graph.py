"""Street graph representation."""

from typing import Optional

import networkx as nx

from .config import CONFIG
from .models import Segment
from .geo import haversine_distance


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
