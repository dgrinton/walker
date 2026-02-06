"""Route planning with novelty prioritization."""

from typing import Optional

import networkx as nx

from .config import CONFIG
from .models import Segment, RecentPathContext
from .geo import haversine_distance, bearing_between, bearing_to_compass, relative_direction, is_opposite_direction
from .graph import StreetGraph
from .history import HistoryDB


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
        recent_context = RecentPathContext.empty()  # Track recent segments for backtrack detection

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
            scored = [(n, self.score_edge(current, n, used_segments, recent_context)) for n in valid_options]
            scored.sort(key=lambda x: x[1])

            # Pick best option
            next_node = scored[0][0]

            # Record segment
            segment = self.graph.get_segment(current, next_node)
            if segment:
                distance += segment.length
                used_segments.add(segment.id)
                # Track this segment in recent context for backtrack detection
                recent_context.add_segment(
                    segment, current, next_node,
                    CONFIG["recent_segment_history"]
                )

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
                   used_segments: Optional[set[str]] = None,
                   recent_context: Optional[RecentPathContext] = None) -> float:
        """Score an edge - lower is better.

        Considers:
        - Novelty (never walked = 0, walked before = higher)
        - Road type preference
        - Whether it helps complete a loop
        - Segments already used in the current route calculation
        - Backtracking penalty (going back on same/parallel street)
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

        # Backtrack penalty - penalize going back on same/parallel streets
        backtrack_penalty = 0
        if recent_context and recent_context.recent_segments:
            for recent_seg in recent_context.recent_segments:
                # Only penalize if going OPPOSITE direction (not continuing forward)
                if is_opposite_direction(
                    self.graph, from_node, to_node, recent_seg,
                    CONFIG["parallel_angle_threshold"],
                    CONFIG["parallel_distance_threshold"]
                ):
                    # Same street name + opposite direction = definite backtrack
                    if segment.name and segment.name == recent_seg.segment.name:
                        backtrack_penalty += CONFIG["same_street_penalty"]
                    # Opposite direction + close proximity = likely parallel path
                    backtrack_penalty += CONFIG["parallel_segment_penalty"]
                    break  # One backtrack detection is enough

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

        return road_weight + novelty_factor + backtrack_penalty

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
