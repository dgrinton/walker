"""Street graph representation."""

import hashlib
import json
import os
import time
from typing import Optional

import networkx as nx

from .config import CONFIG
from .models import Segment
from .geo import haversine_distance, bearing_between, are_bearings_parallel, point_to_segment_distance, _segments_intersect

GRAPH_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "osm_cache")
GRAPH_CACHE_MAX_AGE = 7 * 24 * 3600  # 7 days


class StreetGraph:
    """Graph representation of street network"""

    def __init__(self):
        self.graph = nx.Graph()
        self.nodes: dict[int, tuple[float, float]] = {}  # node_id -> (lat, lon)
        self.segments: dict[str, Segment] = {}  # segment_id -> Segment
        self.parallel_segments: dict[str, set[str]] = {}  # segment_id -> set of parallel segment_ids
        self.corridor_groups: dict[str, int] = {}  # segment_id -> corridor group_id
        self.corridor_members: dict[int, set[str]] = {}  # group_id -> set of segment_ids

    @staticmethod
    def _osm_cache_key(osm_data: dict) -> str:
        """Compute a stable hash for OSM data + relevant config values."""
        # Include config values that affect graph building
        config_keys = [
            "road_weights", "default_road_weight", "busy_road_types",
            "busy_road_proximity_threshold",
            "busy_road_proximity_penalty", "busy_road_crossing_penalty",
            "simplify_max_segment",
            "virtual_edge_max_distance",
        ]
        config_snapshot = {k: CONFIG.get(k) for k in config_keys}
        # Convert sets to sorted lists for JSON serialization
        for k, v in config_snapshot.items():
            if isinstance(v, set):
                config_snapshot[k] = sorted(v)

        elements = osm_data.get("elements", [])
        hasher = hashlib.sha256()
        hasher.update(json.dumps(len(elements), sort_keys=True).encode())
        # Hash element IDs and types — fast proxy for content identity
        for el in elements:
            hasher.update(f"{el.get('type')}:{el.get('id')}".encode())
        hasher.update(json.dumps(config_snapshot, sort_keys=True).encode())
        return hasher.hexdigest()[:16]

    def _try_load_cache(self, cache_key: str) -> bool:
        """Try to load a cached graph. Returns True on success."""
        os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(GRAPH_CACHE_DIR, f"graph_{cache_key}.json")
        if not os.path.exists(cache_path):
            return False
        try:
            age = time.time() - os.path.getmtime(cache_path)
            if age > GRAPH_CACHE_MAX_AGE:
                os.remove(cache_path)
                return False
            with open(cache_path, "r") as f:
                data = json.load(f)
            # Restore nodes
            self.nodes = {int(k): tuple(v) for k, v in data["nodes"].items()}
            # Restore segments
            for sd in data["segments"]:
                seg = Segment(
                    id=sd["id"], node1=sd["node1"], node2=sd["node2"],
                    way_id=sd["way_id"], name=sd.get("name"),
                    road_type=sd["road_type"], length=sd["length"],
                    busy_road_adjacent=sd.get("busy_road_adjacent", False),
                    busy_road_crossing=sd.get("busy_road_crossing", False),
                )
                self.segments[seg.id] = seg
            # Restore graph edges
            for ed in data["edges"]:
                self.graph.add_edge(ed["n1"], ed["n2"], **ed["attrs"])
            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def _save_cache(self, cache_key: str):
        """Save the built graph to disk cache."""
        os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(GRAPH_CACHE_DIR, f"graph_{cache_key}.json")
        data = {
            "nodes": {str(k): list(v) for k, v in self.nodes.items()},
            "segments": [
                {
                    "id": s.id, "node1": s.node1, "node2": s.node2,
                    "way_id": s.way_id, "name": s.name,
                    "road_type": s.road_type, "length": s.length,
                    "busy_road_adjacent": s.busy_road_adjacent,
                    "busy_road_crossing": s.busy_road_crossing,
                }
                for s in self.segments.values()
            ],
            "edges": [
                {"n1": u, "n2": v, "attrs": dict(d)}
                for u, v, d in self.graph.edges(data=True)
            ],
        }
        with open(cache_path, "w") as f:
            json.dump(data, f)

    def build_from_osm(self, osm_data: dict):
        """Build graph from OSM data, using disk cache when available."""
        cache_key = self._osm_cache_key(osm_data)

        if self._try_load_cache(cache_key):
            print(f"Loaded graph from cache: {len(self.nodes)} nodes, {len(self.segments)} segments")
            return

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

        self._simplify_degree2_chains()
        if CONFIG.get("virtual_edge_max_distance", 0) > 0:
            self._add_virtual_edges()
        self._flag_busy_road_adjacent()  # After simplification so final segments get correct flags
        print(f"Built graph: {len(self.nodes)} nodes, {len(self.segments)} segments")
        self._save_cache(cache_key)

    def _flag_busy_road_adjacent(self):
        """Flag segments that run alongside or cross busy roads.

        Two flags are set on segments:
        - busy_road_adjacent: midpoint is within proximity threshold of a busy road
        - busy_road_crossing: shares a node with a busy road segment (crossing point)

        Applies to ALL non-busy road types (not just footpaths), since residential
        streets and service roads alongside a highway are also unpleasant.

        Uses a grid-based spatial index to avoid O(n×m) haversine calls.
        """
        busy_road_types = CONFIG["busy_road_types"]
        threshold = CONFIG["busy_road_proximity_threshold"]

        # Grid cell size in degrees — ~threshold meters at mid-latitudes
        cell_size = threshold / 111_320.0

        # Collect midpoints of all busy road segments into a grid
        grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
        busy_road_nodes: set[int] = set()
        for segment in self.segments.values():
            if segment.road_type in busy_road_types:
                busy_road_nodes.add(segment.node1)
                busy_road_nodes.add(segment.node2)
                loc1 = self.nodes.get(segment.node1)
                loc2 = self.nodes.get(segment.node2)
                if loc1 and loc2:
                    mid_lat = (loc1[0] + loc2[0]) / 2
                    mid_lon = (loc1[1] + loc2[1]) / 2
                    ci = int(mid_lat / cell_size)
                    cj = int(mid_lon / cell_size)
                    grid.setdefault((ci, cj), []).append((mid_lat, mid_lon))

        if not grid:
            return

        # Check each non-busy segment for adjacency and crossing
        flagged_adj = 0
        flagged_cross = 0
        for segment in self.segments.values():
            if segment.road_type in busy_road_types:
                continue  # Don't flag busy roads themselves

            # Crossing detection: segment shares a node with a busy road
            if segment.node1 in busy_road_nodes or segment.node2 in busy_road_nodes:
                segment.busy_road_crossing = True
                flagged_cross += 1

            # Proximity detection
            loc1 = self.nodes.get(segment.node1)
            loc2 = self.nodes.get(segment.node2)
            if not loc1 or not loc2:
                continue
            mid_lat = (loc1[0] + loc2[0]) / 2
            mid_lon = (loc1[1] + loc2[1]) / 2
            ci = int(mid_lat / cell_size)
            cj = int(mid_lon / cell_size)
            found = False
            for di in (-1, 0, 1):
                if found:
                    break
                for dj in (-1, 0, 1):
                    cell = grid.get((ci + di, cj + dj))
                    if not cell:
                        continue
                    for busy_mid in cell:
                        if haversine_distance(mid_lat, mid_lon, busy_mid[0], busy_mid[1]) < threshold:
                            segment.busy_road_adjacent = True
                            flagged_adj += 1
                            found = True
                            break
                    if found:
                        break

        if flagged_adj or flagged_cross:
            print(f"Flagged {flagged_adj} segments adjacent to busy roads, {flagged_cross} crossing busy roads")

    def _simplify_degree2_chains(self):
        """Contract degree-2 nodes where both adjacent segments are short.

        A degree-2 node on the graph is not a real intersection — the walker
        has no choice to make there.  When both segments touching such a node
        are short (below simplify_max_segment), the node is removed and the
        two segments are merged into one.  This eliminates micro-steps from
        complex OSM intersection geometry.

        Only merges when:
        - Both segments share the same road_type
        - Both segments are under max length
        - The merged segment would also be under max length

        Uses a queue to avoid restarting full node iteration after each merge.
        """
        from collections import deque

        max_seg = CONFIG["simplify_max_segment"]
        removed = 0

        # Seed the queue with all degree-2 nodes
        queue = deque(
            nid for nid in self.graph.nodes if self.graph.degree(nid) == 2
        )
        in_queue = set(queue)

        while queue:
            node_id = queue.popleft()
            in_queue.discard(node_id)

            if node_id not in self.graph:
                continue
            if self.graph.degree(node_id) != 2:
                continue

            neighbors = list(self.graph.neighbors(node_id))
            n1, n2 = neighbors[0], neighbors[1]

            seg1 = self.get_segment(node_id, n1)
            seg2 = self.get_segment(node_id, n2)
            if not seg1 or not seg2:
                continue
            # Only merge short segments of the same road type and OSM way
            if seg1.length > max_seg or seg2.length > max_seg:
                continue
            if seg1.road_type != seg2.road_type:
                continue
            if seg1.way_id != seg2.way_id:
                continue
            merged_length = seg1.length + seg2.length
            # Don't merge if n1 == n2 (loop)
            if n1 == n2:
                continue
            # Already connected — skip
            if self.graph.has_edge(n1, n2):
                continue

            # Merge: create new segment n1—n2
            merged_name = seg1.name or seg2.name
            merged_way_id = seg1.way_id
            merged_road_type = seg1.road_type

            new_seg_id = Segment.make_id(n1, n2)
            new_segment = Segment(
                id=new_seg_id,
                node1=min(n1, n2),
                node2=max(n1, n2),
                way_id=merged_way_id,
                name=merged_name,
                road_type=merged_road_type,
                length=merged_length,
            )
            self.segments[new_seg_id] = new_segment

            weight = CONFIG["road_weights"].get(merged_road_type, CONFIG["default_road_weight"])
            self.graph.add_edge(n1, n2,
                                segment_id=new_seg_id,
                                length=merged_length,
                                weight=weight * merged_length,
                                road_type=merged_road_type,
                                name=merged_name)

            # Remove old segments and node
            del self.segments[seg1.id]
            del self.segments[seg2.id]
            self.graph.remove_node(node_id)
            del self.nodes[node_id]

            removed += 1

            # After merging, n1 or n2 may now be degree-2 — re-check them
            for neighbor in (n1, n2):
                if neighbor in self.graph and self.graph.degree(neighbor) == 2:
                    if neighbor not in in_queue:
                        queue.append(neighbor)
                        in_queue.add(neighbor)

        if removed:
            print(f"Simplified {removed} degree-2 nodes (max segment {max_seg}m)")

    def _add_virtual_edges(self):
        """Add virtual edges between close nodes that aren't directly connected.

        Bridges gaps of up to virtual_edge_max_distance meters, allowing
        the planner to cross between disconnected footway networks.
        Virtual edges are not added if they would cross a busy road segment.
        """
        max_dist = CONFIG["virtual_edge_max_distance"]
        busy_road_types = CONFIG["busy_road_types"]

        # Collect busy road segments for crossing checks
        busy_segments: list[tuple[float, float, float, float]] = []
        for seg in self.segments.values():
            if seg.road_type in busy_road_types:
                loc1 = self.nodes.get(seg.node1)
                loc2 = self.nodes.get(seg.node2)
                if loc1 and loc2:
                    busy_segments.append((loc1[0], loc1[1], loc2[0], loc2[1]))

        # Sort nodes by latitude for spatial pre-filtering
        sorted_nodes = sorted(self.nodes.items(), key=lambda x: x[1][0])
        lat_per_meter = 1.0 / 111_320
        lat_margin = max_dist * lat_per_meter

        added = 0
        n = len(sorted_nodes)
        for i in range(n):
            n1_id, (lat1, lon1) = sorted_nodes[i]
            for j in range(i + 1, n):
                n2_id, (lat2, lon2) = sorted_nodes[j]
                # Latitude pre-filter
                if lat2 - lat1 > lat_margin:
                    break
                # Skip if already connected
                if self.graph.has_edge(n1_id, n2_id):
                    continue
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                if dist > max_dist or dist < 1:
                    continue
                # Check that virtual edge doesn't cross any busy road
                crosses_busy = False
                for blat1, blon1, blat2, blon2 in busy_segments:
                    if _segments_intersect(lat1, lon1, lat2, lon2,
                                           blat1, blon1, blat2, blon2):
                        crosses_busy = True
                        break
                if crosses_busy:
                    continue

                # Create virtual segment and edge
                seg_id = Segment.make_id(n1_id, n2_id)
                if seg_id in self.segments:
                    continue  # Already exists (shouldn't happen but be safe)
                segment = Segment(
                    id=seg_id,
                    node1=min(n1_id, n2_id),
                    node2=max(n1_id, n2_id),
                    way_id=0,  # Virtual — no OSM way
                    name=None,
                    road_type="virtual",
                    length=dist,
                )
                self.segments[seg_id] = segment
                weight = CONFIG["default_road_weight"]
                self.graph.add_edge(n1_id, n2_id,
                                    segment_id=seg_id,
                                    length=dist,
                                    weight=weight * dist,
                                    road_type="virtual",
                                    name=None)
                added += 1

        if added:
            print(f"Added {added} virtual edges (max {max_dist}m, no busy road crossing)")

    def _compute_parallel_segments(self):
        """Precompute mapping of parallel segments for corridor deduplication.

        Two segments are parallel if they belong to different OSM ways, share no
        nodes, have midpoints within the proximity threshold, and bearings within
        the angle threshold (direction-agnostic).
        """
        proximity = CONFIG["parallel_distance_threshold"]
        angle_threshold = CONFIG["parallel_angle_threshold"]
        min_length = CONFIG["corridor_min_segment_length"]

        # Pre-compute midpoints, bearings, and endpoints for eligible segments
        # (id, seg, mid_lat, mid_lon, bearing, lat1, lon1, lat2, lon2)
        seg_info: list[tuple[str, Segment, float, float, float, float, float, float, float]] = []
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
            seg_info.append((seg.id, seg, mid_lat, mid_lon, seg_bearing,
                             loc1[0], loc1[1], loc2[0], loc2[1]))

        # Sort by latitude for pre-filtering
        seg_info.sort(key=lambda x: x[2])

        # Approximate latitude degrees per meter (for pre-filter)
        lat_per_meter = 1.0 / 111_320
        lat_margin = proximity * lat_per_meter

        pair_count = 0
        n = len(seg_info)
        for i in range(n):
            id_a, seg_a, lat_a, lon_a, bearing_a, a1lat, a1lon, a2lat, a2lon = seg_info[i]
            nodes_a = {seg_a.node1, seg_a.node2}
            for j in range(i + 1, n):
                id_b, seg_b, lat_b, lon_b, bearing_b, b1lat, b1lon, b2lat, b2lon = seg_info[j]
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
                # Check proximity: midpoint of each segment to the other's line
                dist_a_to_b = point_to_segment_distance(lat_a, lon_a, b1lat, b1lon, b2lat, b2lon)
                dist_b_to_a = point_to_segment_distance(lat_b, lon_b, a1lat, a1lon, a2lat, a2lon)
                if min(dist_a_to_b, dist_b_to_a) >= proximity:
                    continue
                # Record bidirectional mapping
                self.parallel_segments.setdefault(id_a, set()).add(id_b)
                self.parallel_segments.setdefault(id_b, set()).add(id_a)
                pair_count += 1

        if pair_count:
            print(f"Detected {pair_count} parallel segment pairs for corridor deduplication")

    def _compute_name_corridors(self):
        """Group same-name segments into corridor groups based on proximity.

        Segments with the same street name whose midpoints are within the
        corridor proximity threshold are grouped together.  When any segment
        in a group is walked, the whole group can be marked as used to prevent
        the planner from zigzagging between parallel carriageways / footpaths
        that share the same name.
        """
        from collections import defaultdict

        proximity = CONFIG["corridor_name_proximity"]

        # Group segments by name
        name_to_segs: dict[str, list[tuple[Segment, float, float]]] = defaultdict(list)
        for seg in self.segments.values():
            if not seg.name:
                continue
            loc1 = self.nodes.get(seg.node1)
            loc2 = self.nodes.get(seg.node2)
            if not loc1 or not loc2:
                continue
            mid_lat = (loc1[0] + loc2[0]) / 2
            mid_lon = (loc1[1] + loc2[1]) / 2
            name_to_segs[seg.name].append((seg, mid_lat, mid_lon))

        group_id = 0
        for name, entries in name_to_segs.items():
            if len(entries) < 2:
                continue

            # Build adjacency: two same-name segments are connected if
            # midpoints are within proximity threshold
            n = len(entries)
            adj: list[list[int]] = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    dist = haversine_distance(
                        entries[i][1], entries[i][2],
                        entries[j][1], entries[j][2],
                    )
                    if dist < proximity:
                        adj[i].append(j)
                        adj[j].append(i)

            # BFS for connected components
            visited = [False] * n
            for start in range(n):
                if visited[start]:
                    continue
                component: list[int] = []
                queue = [start]
                while queue:
                    idx = queue.pop()
                    if visited[idx]:
                        continue
                    visited[idx] = True
                    component.append(idx)
                    for neighbor in adj[idx]:
                        if not visited[neighbor]:
                            queue.append(neighbor)

                if len(component) >= 2:
                    seg_ids = {entries[i][0].id for i in component}
                    self.corridor_members[group_id] = seg_ids
                    for sid in seg_ids:
                        self.corridor_groups[sid] = group_id
                    group_id += 1

        if self.corridor_members:
            total_segs = sum(len(m) for m in self.corridor_members.values())
            print(f"Computed {len(self.corridor_members)} name-based corridor groups ({total_segs} segments)")

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
