"""Route planning with novelty prioritization."""

from typing import Optional

import networkx as nx

from .config import CONFIG
from .models import Segment
from .geo import haversine_distance, bearing_between, bearing_to_compass, relative_direction, point_in_any_polygon, segment_crosses_any_polygon, segment_buffer_polygon
from .graph import StreetGraph
from .history import HistoryDB


class RoutePlanner:
    """Plans routes prioritizing novelty"""

    def __init__(self, graph: StreetGraph, history: HistoryDB,
                 excluded_zones: Optional[list[list[list[float]]]] = None):
        self.graph = graph
        self.history = history
        self.start_node: Optional[int] = None
        self.target_distance: float = 0
        self.walked_distance: float = 0
        self.current_path: list[int] = []
        self.planned_route: list[int] = []  # Full pre-calculated route
        self.route_index: int = 0  # Current position in planned_route
        self.excluded_nodes: set[int] = set()
        self._allowed_graph = self.graph.graph

        if excluded_zones:
            self._compute_excluded_nodes(excluded_zones)

    def _compute_excluded_nodes(self, excluded_zones: list[list[list[float]]]):
        """Pre-compute which nodes and edges are blocked by exclusion zones."""
        for node_id in self.graph.graph.nodes:
            loc = self.graph.get_node_location(node_id)
            if loc and point_in_any_polygon(loc[0], loc[1], excluded_zones):
                self.excluded_nodes.add(node_id)

        # Find edges that cross through zones even if neither endpoint is inside
        excluded_edges: set[tuple[int, int]] = set()
        for u, v in self.graph.graph.edges:
            if u in self.excluded_nodes or v in self.excluded_nodes:
                continue  # Already handled by node exclusion
            loc_u = self.graph.get_node_location(u)
            loc_v = self.graph.get_node_location(v)
            if loc_u and loc_v and segment_crosses_any_polygon(
                loc_u[0], loc_u[1], loc_v[0], loc_v[1], excluded_zones
            ):
                excluded_edges.add((u, v))

        if self.excluded_nodes or excluded_edges:
            allowed_nodes = [n for n in self.graph.graph.nodes if n not in self.excluded_nodes]
            self._allowed_graph = self.graph.graph.subgraph(allowed_nodes).copy()
            for u, v in excluded_edges:
                if self._allowed_graph.has_edge(u, v):
                    self._allowed_graph.remove_edge(u, v)

    def _is_dead_end_chain(self, node: int, from_node: int) -> bool:
        """Check if entering node leads to a dead end (no intersection) within a few steps."""
        current = node
        prev = from_node
        for _ in range(CONFIG["dead_end_lookahead"]):
            if current not in self._allowed_graph:
                return True
            neighbors = [n for n in self._allowed_graph.neighbors(current) if n != prev]
            if len(neighbors) == 0:
                return True  # Dead end
            if len(neighbors) > 1:
                return False  # Intersection — not a dead-end chain
            prev = current
            current = neighbors[0]
        return False

    def _mark_segment_used(self, segment: Segment, used_segments: set[str]):
        """Mark a segment as used and create a buffer polygon around it."""
        used_segments.add(segment.id)
        loc1 = self.graph.get_node_location(segment.node1)
        loc2 = self.graph.get_node_location(segment.node2)
        if loc1 and loc2:
            poly = segment_buffer_polygon(
                loc1[0], loc1[1], loc2[0], loc2[1],
                width=CONFIG["walk_buffer_width"],
                tip_angle=CONFIG["walk_buffer_tip_angle"],
                end_inset=CONFIG["walk_buffer_end_inset"],
                min_length=CONFIG["walk_buffer_min_length"],
            )
            if poly is not None:
                self._walk_buffers.append(poly)

    def _return_path_weight(self, used_segments: set[str], u: int, v: int, data: dict) -> float:
        """Weight function for return path that penalizes already-used segments."""
        length = data.get("length", 1)
        seg_id = Segment.make_id(u, v)
        if seg_id in used_segments:
            return length * 10
        return length

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

        # Ensure start node is never excluded
        self.excluded_nodes.discard(start_node)

        route = [start_node]
        current = start_node
        previous = None
        distance = 0
        used_segments: set[str] = set()  # Track segments used in this route
        self._walk_buffers: list[list[list[float]]] = []  # Buffer polygons around walked segments

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
                # Find path back to start (penalizing already-used segments)
                try:
                    path_home = nx.shortest_path(
                        self._allowed_graph, current, start_node,
                        weight=lambda u, v, d: self._return_path_weight(used_segments, u, v, d)
                    )
                    # Add remaining path (excluding current which is already in route)
                    for node in path_home[1:]:
                        segment = self.graph.get_segment(route[-1], node)
                        if segment:
                            distance += segment.length
                            self._mark_segment_used(segment, used_segments)
                        route.append(node)
                    break
                except nx.NetworkXNoPath:
                    break

            # Get neighbors from allowed graph (respects exclusion zones)
            if current not in self._allowed_graph:
                break
            neighbors = list(self._allowed_graph.neighbors(current))
            if not neighbors:
                break

            # Filter out where we came from (unless it's our only option)
            options = [n for n in neighbors if n != previous]
            if not options:
                options = neighbors

            # Filter options that would take us too far to return or cross walk buffers
            valid_options = []
            for n in options:
                n_loc = self.graph.get_node_location(n)
                if not n_loc:
                    continue
                # Skip if segment crosses any walk buffer polygon
                if current_loc and self._walk_buffers:
                    if segment_crosses_any_polygon(
                        current_loc[0], current_loc[1],
                        n_loc[0], n_loc[1],
                        self._walk_buffers,
                    ):
                        continue
                segment = self.graph.get_segment(current, n)
                seg_len = segment.length if segment else 0
                n_dist_to_start = haversine_distance(
                    n_loc[0], n_loc[1], start_loc[0], start_loc[1]
                )
                new_remaining = remaining_budget - seg_len
                if new_remaining > n_dist_to_start * 0.9:
                    valid_options.append(n)

            if not valid_options:
                break  # for debugging
                # No valid options, try to go home (penalizing already-used segments)
                try:
                    path_home = nx.shortest_path(
                        self._allowed_graph, current, start_node,
                        weight=lambda u, v, d: self._return_path_weight(used_segments, u, v, d)
                    )
                    for node in path_home[1:]:
                        segment = self.graph.get_segment(route[-1], node)
                        if segment:
                            distance += segment.length
                            self._mark_segment_used(segment, used_segments)
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
                self._mark_segment_used(segment, used_segments)

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

        # Busy road proximity penalty for footpaths alongside busy roads
        busy_road_penalty = CONFIG["busy_road_proximity_penalty"] if segment.busy_road_adjacent else 0

        # Dead-end penalty — avoid entering chains that lead to dead ends
        dead_end_penalty = CONFIG["dead_end_penalty"] if self._is_dead_end_chain(to_node, from_node) else 0

        return road_weight + novelty_factor + busy_road_penalty + dead_end_penalty

    def choose_next_direction(self, current_node: int, came_from: Optional[int] = None) -> Optional[int]:
        """Choose the best next node to walk to"""
        if current_node not in self._allowed_graph:
            return None
        neighbors = list(self._allowed_graph.neighbors(current_node))

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
                        self._allowed_graph, current_node, self.start_node, weight="length"
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
