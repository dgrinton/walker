"""Route planning with novelty prioritization."""

import math
from typing import Optional

import networkx as nx

from .config import CONFIG
from .models import Segment
from .geo import haversine_distance, bearing_between, bearing_to_compass, relative_direction, point_in_any_polygon, segment_crosses_any_polygon, segment_buffer_polygon
from .graph import StreetGraph
from .history import HistoryDB
from .route_log import RouteLogger


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
        self._route_logging_enabled = False
        self._route_log_path: Optional[str] = None
        self._route_logger: Optional[RouteLogger] = None
        self._walk_buffers: list[list[list[float]]] = []
        self._walked_midpoints: list[tuple[float, float, float]] = []  # (lat, lon, bearing) for parallel detection

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
        """Check if entering node leads to a dead end (no intersection) within a few steps.

        Also considers walk buffer polygons — edges that cross a buffer are
        treated as blocked, so service-road clusters with buffer-blocked exits
        are correctly detected as dead ends.
        """
        current = node
        prev = from_node
        for _ in range(CONFIG["dead_end_lookahead"]):
            if current not in self._allowed_graph:
                return True
            current_loc = self.graph.get_node_location(current)
            neighbors = []
            for n in self._allowed_graph.neighbors(current):
                if n == prev:
                    continue
                # Check if this edge is blocked by walk buffers
                if self._walk_buffers and current_loc:
                    n_loc = self.graph.get_node_location(n)
                    if n_loc and segment_crosses_any_polygon(
                        current_loc[0], current_loc[1],
                        n_loc[0], n_loc[1],
                        self._walk_buffers,
                    ):
                        continue
                neighbors.append(n)
            if len(neighbors) == 0:
                return True  # Dead end
            if len(neighbors) > 1:
                return False  # Intersection — not a dead-end chain
            prev = current
            current = neighbors[0]
        return False

    def _mark_segment_used(self, segment: Segment, used_segments: set[str]):
        """Mark a segment as used and create a buffer polygon around it.

        Also records the segment midpoint and bearing for parallel-segment
        detection in score_edge.
        """
        used_segments.add(segment.id)
        loc1 = self.graph.get_node_location(segment.node1)
        loc2 = self.graph.get_node_location(segment.node2)
        if loc1 and loc2:
            buf_width = CONFIG["walk_buffer_width"]
            poly = segment_buffer_polygon(
                loc1[0], loc1[1], loc2[0], loc2[1],
                width=buf_width,
                tip_angle=CONFIG["walk_buffer_tip_angle"],
                end_inset=CONFIG["walk_buffer_end_inset"],
                min_length=CONFIG["walk_buffer_min_length"],
            )
            if poly is not None:
                self._walk_buffers.append(poly)
            # Record midpoint + bearing for parallel-segment detection
            # Only record substantial segments to avoid noise from dense short segments
            if segment.length >= 30:
                mid_lat = (loc1[0] + loc2[0]) / 2
                mid_lon = (loc1[1] + loc2[1]) / 2
                seg_bearing = bearing_between(loc1[0], loc1[1], loc2[0], loc2[1])
                self._walked_midpoints.append((mid_lat, mid_lon, seg_bearing))

    def _return_path_weight(self, used_segments: set[str], u: int, v: int, data: dict) -> float:
        """Weight function for return path — prefer novel segments, avoid busy roads.

        Heavily penalizes segments already used in this route, and moderately
        penalizes historically walked segments. Novel segments are at base cost.
        Also penalizes busy-road-adjacent and crossing segments.
        """
        length = data.get("length", 1)
        seg_id = Segment.make_id(u, v)
        base = length
        # Busy road penalty on return path too
        seg = self.graph.segments.get(seg_id)
        if seg:
            if seg.road_type in CONFIG["busy_road_types"] or seg.road_type == "motorway":
                # Walking ON a busy road — strongest penalty
                base = length * 8
            elif seg.busy_road_crossing:
                base = length * 5
            elif seg.busy_road_adjacent:
                base = length * 3
        if seg_id in used_segments:
            return base * 20  # Strong penalty — force detour rather than retrace
        times_walked, _ = self.history.get_segment_history(seg_id)
        if times_walked == 0:
            return base  # Base cost for novel segments
        return base * (1 + times_walked * 0.5)  # Mild penalty for walked segments

    def enable_route_logging(self, path: str):
        """Enable route logging. Must be called before calculate_full_route()."""
        self._route_logging_enabled = True
        self._route_log_path = path

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
        When the planner gets trapped (no valid options), it backtracks to the
        last junction and tries a different direction.
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
        self.walked_distance = 0  # Keep in sync for score_edge
        self._initial_bearing: Optional[float] = None  # Set after first substantial step
        used_segments: set[str] = set()  # Track segments used in this route
        self._walk_buffers: list[list[list[float]]] = []  # Buffer polygons around walked segments
        self._walked_midpoints: list[tuple[float, float, float]] = []  # (lat, lon, bearing) for parallel detection
        visited_nodes: dict[int, int] = {start_node: 1}  # node_id -> visit count
        step_index = 0
        prev_bearing: Optional[float] = None
        consecutive_busy_adj = 0  # Track consecutive busy-road-adjacent steps

        # Backtracking state
        # blacklisted_edges: directed edges (from, to) that led to traps
        blacklisted_edges: set[tuple[int, int]] = set()
        # Snapshots at junction nodes for efficient rollback
        # Each entry: (route_idx, distance, walked_distance, used_segments_copy,
        #              walk_buffers_copy, visited_nodes_copy, previous, step_index,
        #              prev_bearing, initial_bearing, consecutive_busy_adj, walked_midpoints_copy)
        junction_snapshots: list[tuple] = []
        max_backtracks = 20  # Safety limit
        backtrack_count = 0

        # Set up route logging if enabled
        logging = self._route_logging_enabled
        logger: Optional[RouteLogger] = None
        if logging:
            start_loc = self.graph.get_node_location(start_node)
            if start_loc:
                logger = RouteLogger(
                    start_node=start_node,
                    start_location=start_loc,
                    target_distance=target_distance,
                    graph_stats={
                        "total_nodes": len(self.graph.nodes),
                        "total_segments": len(self.graph.segments),
                    },
                )

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

            # Head home when: enough distance walked AND not enough budget for scenic return
            if distance >= target_distance * 0.5 and remaining_budget < dist_to_start * 1.5:
                # Find path back to start (penalizing already-used segments)
                try:
                    path_home = nx.shortest_path(
                        self._allowed_graph, current, start_node,
                        weight=lambda u, v, d: self._return_path_weight(used_segments, u, v, d)
                    )
                    return_distance = 0
                    # Add remaining path (excluding current which is already in route)
                    for node in path_home[1:]:
                        segment = self.graph.get_segment(route[-1], node)
                        if segment:
                            distance += segment.length
                            return_distance += segment.length
                            self._mark_segment_used(segment, used_segments)
                        route.append(node)
                    if logger:
                        return_locs = [self.graph.get_node_location(n) for n in path_home]
                        return_locs = [loc for loc in return_locs if loc]
                        logger.log_return_path(
                            trigger="budget_threshold",
                            triggered_at_step=step_index,
                            return_nodes=path_home,
                            return_distance=return_distance,
                            return_locations=return_locs,
                        )
                    break
                except nx.NetworkXNoPath:
                    break

            # Get neighbors from allowed graph (respects exclusion zones)
            if current not in self._allowed_graph:
                break
            neighbors = list(self._allowed_graph.neighbors(current))
            if not neighbors:
                break

            # Filter out where we came from and blacklisted edges
            options = [n for n in neighbors if n != previous and (current, n) not in blacklisted_edges]
            if not options:
                options = [n for n in neighbors if (current, n) not in blacklisted_edges]
            if not options:
                options = neighbors  # Last resort: allow any direction

            # Filter options that would take us too far to return or cross walk buffers
            valid_options = []
            rejected: list[dict] = []  # For logging
            for n in options:
                n_loc = self.graph.get_node_location(n)
                if not n_loc:
                    if logging:
                        seg = self.graph.get_segment(current, n)
                        rejected.append({
                            "to_node": n,
                            "segment_id": seg.id if seg else Segment.make_id(current, n),
                            "segment_name": seg.name if seg else None,
                            "road_type": seg.road_type if seg else None,
                            "score": None,
                            "score_breakdown": None,
                            "rejection_reason": "no_location",
                        })
                    continue
                # Skip if segment crosses any walk buffer polygon
                if current_loc and self._walk_buffers:
                    if segment_crosses_any_polygon(
                        current_loc[0], current_loc[1],
                        n_loc[0], n_loc[1],
                        self._walk_buffers,
                    ):
                        if logging:
                            seg = self.graph.get_segment(current, n)
                            rejected.append({
                                "to_node": n,
                                "segment_id": seg.id if seg else Segment.make_id(current, n),
                                "segment_name": seg.name if seg else None,
                                "road_type": seg.road_type if seg else None,
                                "score": None,
                                "score_breakdown": None,
                                "rejection_reason": "buffer_crossing",
                            })
                        continue
                segment = self.graph.get_segment(current, n)
                seg_len = segment.length if segment else 0
                n_dist_to_start = haversine_distance(
                    n_loc[0], n_loc[1], start_loc[0], start_loc[1]
                )
                new_remaining = remaining_budget - seg_len
                if new_remaining > n_dist_to_start * 0.9:
                    valid_options.append(n)
                elif logging:
                    rejected.append({
                        "to_node": n,
                        "segment_id": segment.id if segment else Segment.make_id(current, n),
                        "segment_name": segment.name if segment else None,
                        "road_type": segment.road_type if segment else None,
                        "score": None,
                        "score_breakdown": None,
                        "rejection_reason": "budget_exceeded",
                    })

            if not valid_options:
                # Try backtracking to the last junction
                if junction_snapshots and backtrack_count < max_backtracks:
                    # Blacklist the edge from the junction that led into this dead branch
                    snap = junction_snapshots[-1]
                    snap_route_idx = snap[0]
                    junction_node = route[snap_route_idx]
                    # The edge leaving the junction is route[snap_route_idx] -> route[snap_route_idx + 1]
                    if snap_route_idx + 1 < len(route):
                        bad_next = route[snap_route_idx + 1]
                        blacklisted_edges.add((junction_node, bad_next))

                    # Restore state from snapshot
                    (_, distance, self.walked_distance, used_segments,
                     self._walk_buffers, visited_nodes, previous,
                     step_index, prev_bearing, self._initial_bearing,
                     consecutive_busy_adj, self._walked_midpoints) = snap
                    used_segments = set(used_segments)  # Defensive copy
                    self._walk_buffers = list(self._walk_buffers)
                    self._walked_midpoints = list(self._walked_midpoints)
                    visited_nodes = dict(visited_nodes)
                    route = route[:snap_route_idx + 1]
                    current = junction_node

                    # Trim logger steps back to snapshot
                    if logger:
                        logger.data["steps"] = logger.data["steps"][:step_index]
                        logger._cumulative_distance = distance

                    backtrack_count += 1
                    # Remove this snapshot — if we get trapped again we'll go further back
                    junction_snapshots.pop()
                    continue

                # No snapshots left or too many backtracks — go home
                try:
                    path_home = nx.shortest_path(
                        self._allowed_graph, current, start_node,
                        weight=lambda u, v, d: self._return_path_weight(used_segments, u, v, d)
                    )
                    return_distance = 0
                    for node in path_home[1:]:
                        segment = self.graph.get_segment(route[-1], node)
                        if segment:
                            distance += segment.length
                            return_distance += segment.length
                            self._mark_segment_used(segment, used_segments)
                        route.append(node)
                    if logger:
                        return_locs = [self.graph.get_node_location(n) for n in path_home]
                        return_locs = [loc for loc in return_locs if loc]
                        logger.log_return_path(
                            trigger="no_valid_options",
                            triggered_at_step=step_index,
                            return_nodes=path_home,
                            return_distance=return_distance,
                            return_locations=return_locs,
                        )
                except nx.NetworkXNoPath:
                    pass
                break

            # Save snapshot at junction nodes (multiple valid options)
            if len(valid_options) > 1:
                junction_snapshots.append((
                    len(route) - 1,  # route index of current node
                    distance, self.walked_distance,
                    set(used_segments), list(self._walk_buffers),
                    dict(visited_nodes), previous,
                    step_index, prev_bearing, self._initial_bearing,
                    consecutive_busy_adj, list(self._walked_midpoints),
                ))

            # Sync walked_distance for score_edge (convexity bias)
            self.walked_distance = distance

            # Score all valid options and pick the best
            if logging:
                scored_with_breakdown = []
                for n in valid_options:
                    score, breakdown = self.score_edge(current, n, used_segments, return_breakdown=True, visited_nodes=visited_nodes)
                    scored_with_breakdown.append((n, score, breakdown))
                scored_with_breakdown.sort(key=lambda x: x[1])

                # Build the simple scored list for choosing
                scored = [(n, s) for n, s, _ in scored_with_breakdown]
            else:
                scored = [(n, self.score_edge(current, n, used_segments, visited_nodes=visited_nodes)) for n in valid_options]
                scored.sort(key=lambda x: x[1])

            # Pick best option
            next_node = scored[0][0]
            chosen_score = scored[0][1]

            # Backtrack if the best option is terrible (busy road, crossing penalty, dead end)
            # This catches traps before committing to them
            best_seg = self.graph.get_segment(current, next_node)
            should_backtrack = False
            if best_seg and junction_snapshots and backtrack_count < max_backtracks:
                # Trigger on any busy road type (secondary, primary, trunk, motorway)
                if best_seg.road_type in CONFIG["busy_road_types"] or best_seg.road_type == "motorway":
                    should_backtrack = True
                # Also trigger on busy road crossing (shared node with busy road)
                elif best_seg.busy_road_crossing:
                    should_backtrack = True
                # Trigger when entering a busy-adjacent corridor (3+ consecutive steps)
                elif best_seg.busy_road_adjacent and consecutive_busy_adj >= 2:
                    should_backtrack = True
            if should_backtrack:
                # Blacklist the edge we were about to take from current node
                blacklisted_edges.add((current, next_node))
                # Restore state from the last junction snapshot
                snap = junction_snapshots[-1]
                snap_route_idx = snap[0]
                junction_node = route[snap_route_idx]
                # Also blacklist the edge that led from the junction into this corridor
                if snap_route_idx + 1 < len(route):
                    bad_next = route[snap_route_idx + 1]
                    blacklisted_edges.add((junction_node, bad_next))
                (_, distance, self.walked_distance, used_segments,
                 self._walk_buffers, visited_nodes, previous,
                 step_index, prev_bearing, self._initial_bearing,
                 consecutive_busy_adj, self._walked_midpoints) = snap
                used_segments = set(used_segments)
                self._walk_buffers = list(self._walk_buffers)
                self._walked_midpoints = list(self._walked_midpoints)
                visited_nodes = dict(visited_nodes)
                route = route[:snap_route_idx + 1]
                current = junction_node
                if logger:
                    logger.data["steps"] = logger.data["steps"][:step_index]
                    logger._cumulative_distance = distance
                backtrack_count += 1
                junction_snapshots.pop()
                continue

            # Record segment
            segment = self.graph.get_segment(current, next_node)
            if segment:
                distance += segment.length
                self._mark_segment_used(segment, used_segments)
                # Track consecutive busy-road-adjacent steps
                if segment.busy_road_adjacent:
                    consecutive_busy_adj += 1
                else:
                    consecutive_busy_adj = 0

            # Capture initial bearing from first substantial move (>20m from start)
            if self._initial_bearing is None and distance > 20:
                next_loc_tmp = self.graph.get_node_location(next_node)
                if next_loc_tmp and start_loc:
                    self._initial_bearing = bearing_between(
                        start_loc[0], start_loc[1],
                        next_loc_tmp[0], next_loc_tmp[1],
                    )

            # Log this step
            if logger and segment:
                next_loc = self.graph.get_node_location(next_node)
                cur_bearing = bearing_between(
                    current_loc[0], current_loc[1],
                    next_loc[0], next_loc[1],
                ) if next_loc else 0
                turn_angle = None
                if prev_bearing is not None:
                    diff = cur_bearing - prev_bearing
                    turn_angle = ((diff + 180) % 360) - 180

                # Build alternatives list (non-chosen valid options + rejected)
                alternatives = []
                if logging:
                    for n, s, bd in scored_with_breakdown[1:]:
                        alt_seg = self.graph.get_segment(current, n)
                        alternatives.append({
                            "to_node": n,
                            "segment_id": alt_seg.id if alt_seg else Segment.make_id(current, n),
                            "segment_name": alt_seg.name if alt_seg else None,
                            "road_type": alt_seg.road_type if alt_seg else None,
                            "score": round(s, 2) if s != float("inf") else "inf",
                            "score_breakdown": {k: round(v, 2) if isinstance(v, float) else v
                                                for k, v in bd.items()},
                            "rejection_reason": None,
                        })
                alternatives.extend(rejected)

                # Get chosen breakdown
                chosen_breakdown = scored_with_breakdown[0][2] if logging else {}

                logger.log_step(
                    step_index=step_index,
                    from_node=current,
                    to_node=next_node,
                    segment_id=segment.id,
                    segment_name=segment.name,
                    road_type=segment.road_type,
                    segment_length=segment.length,
                    from_location=current_loc,
                    to_location=next_loc,
                    bearing=cur_bearing,
                    turn_angle=turn_angle,
                    busy_road_adjacent=segment.busy_road_adjacent,
                    phase="explore",
                    chosen_score=chosen_score,
                    score_breakdown=chosen_breakdown,
                    alternatives=alternatives,
                    dist_to_start=dist_to_start,
                    remaining_budget=remaining_budget,
                    walk_buffers_count=len(self._walk_buffers),
                )
                prev_bearing = cur_bearing
                step_index += 1

            route.append(next_node)
            visited_nodes[next_node] = visited_nodes.get(next_node, 0) + 1
            previous = current
            current = next_node

            # Safety check for infinite loops
            if len(route) > 10000:
                break

        # Save route log if enabled
        if logger:
            logger.finalize(distance)
            if self._route_log_path:
                logger.save(self._route_log_path)
            self._route_logger = logger

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
                   return_breakdown: bool = False,
                   visited_nodes: Optional[dict[int, int]] = None):
        """Score an edge - lower is better.

        Considers:
        - Novelty (never walked = 0, walked before = higher)
        - Road type preference
        - Whether it helps complete a loop
        - Segments already used in the current route calculation
        - Whether the target node has already been visited (trap avoidance)

        If return_breakdown is True, returns (score, breakdown_dict).
        """
        segment = self.graph.get_segment(from_node, to_node)
        if not segment:
            if return_breakdown:
                return float("inf"), {"error": "no_segment"}
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
                if return_breakdown:
                    return float("inf"), {
                        "road_weight": road_weight,
                        "novelty_factor": novelty_factor,
                        "busy_road_penalty": 0,
                        "dead_end_penalty": 0,
                        "over_budget": True,
                    }
                return float("inf")  # Can't return in budget

        # Busy road penalties — crossing is much worse than adjacent
        busy_road_penalty = 0
        if segment.busy_road_crossing:
            busy_road_penalty = CONFIG["busy_road_crossing_penalty"]
        elif segment.busy_road_adjacent:
            busy_road_penalty = CONFIG["busy_road_proximity_penalty"]

        # Dead-end penalty — avoid entering chains that lead to dead ends
        dead_end_penalty = CONFIG["dead_end_penalty"] if self._is_dead_end_chain(to_node, from_node) else 0

        # Loop steering — constant curvature to form a circuit.
        # Two components:
        # 1. Bearing: ideal bearing sweeps 360° over the walk
        # 2. Distance: penalize being too far from start (beyond ideal loop radius)
        convexity_penalty = 0
        progress = self.walked_distance / self.target_distance if self.target_distance > 0 else 0
        if self._initial_bearing is not None and to_loc and start_loc:
            from_loc = self.graph.get_node_location(from_node)
            if from_loc:
                edge_bearing = bearing_between(
                    from_loc[0], from_loc[1], to_loc[0], to_loc[1]
                )
                ideal_bearing = (self._initial_bearing + progress * 360) % 360
                # Angular deviation (0-180)
                deviation = abs(((edge_bearing - ideal_bearing + 180) % 360) - 180)
                deviation_norm = deviation / 180.0
                bearing_penalty = deviation_norm * CONFIG["loop_steering_weight"]

                # Distance penalty: penalize being further than the ideal loop radius
                # Ideal circle: circumference = target_distance, radius = target / (2*pi)
                ideal_radius = self.target_distance / (2 * math.pi)
                to_dist = haversine_distance(
                    to_loc[0], to_loc[1], start_loc[0], start_loc[1]
                )
                excess = max(0, to_dist - ideal_radius)
                dist_penalty = excess * CONFIG["loop_steering_dist_weight"]

                convexity_penalty = bearing_penalty + dist_penalty

        # Visited-node penalty — discourage revisiting nodes already in the route
        visited_penalty = 0
        if visited_nodes and to_node in visited_nodes:
            visit_count = visited_nodes[to_node]
            visited_penalty = CONFIG["visited_node_penalty"] * visit_count

        # Parallel-segment penalty — penalize edges that run parallel or opposite
        # to previously walked segments within a proximity threshold.
        # This catches cases where buffers are too narrow (e.g. parallel footways
        # 15-20m apart that the buffer hexagon doesn't span).
        parallel_penalty = 0
        if self._walked_midpoints and to_loc:
            from_loc_pp = self.graph.get_node_location(from_node)
            if from_loc_pp:
                edge_bearing = bearing_between(
                    from_loc_pp[0], from_loc_pp[1], to_loc[0], to_loc[1]
                )
                edge_mid_lat = (from_loc_pp[0] + to_loc[0]) / 2
                edge_mid_lon = (from_loc_pp[1] + to_loc[1]) / 2
                proximity = CONFIG["parallel_segment_proximity"]
                for wlat, wlon, wbear in self._walked_midpoints:
                    dist = haversine_distance(edge_mid_lat, edge_mid_lon, wlat, wlon)
                    if dist > proximity:
                        continue
                    # Check bearing similarity (parallel or opposite)
                    bear_diff = abs(((edge_bearing - wbear + 180) % 360) - 180)
                    if bear_diff < 30 or bear_diff > 150:
                        parallel_penalty = CONFIG["parallel_segment_penalty"]
                        break

        score = road_weight + novelty_factor + busy_road_penalty + dead_end_penalty + convexity_penalty + visited_penalty + parallel_penalty

        if return_breakdown:
            return score, {
                "road_weight": road_weight,
                "novelty_factor": novelty_factor,
                "busy_road_penalty": busy_road_penalty,
                "dead_end_penalty": dead_end_penalty,
                "convexity_penalty": round(convexity_penalty, 2),
                "visited_penalty": visited_penalty,
                "parallel_penalty": parallel_penalty,
            }
        return score

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
