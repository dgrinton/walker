"""Route logging and evaluation for iterative route improvement."""

import json
import math
from datetime import datetime, timezone
from typing import Optional

from .config import CONFIG
from .geo import haversine_distance


class RouteLogger:
    """Records per-step decisions during route planning for analysis."""

    def __init__(self, start_node: int, start_location: tuple[float, float],
                 target_distance: float, graph_stats: dict,
                 iteration: int = 0):
        self.data = {
            "version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "config": {k: v if not isinstance(v, set) else sorted(v)
                       for k, v in CONFIG.items()},
            "start_node": start_node,
            "start_location": {"lat": start_location[0], "lon": start_location[1]},
            "target_distance_m": target_distance,
            "actual_distance_m": 0,
            "graph_stats": graph_stats,
            "steps": [],
            "return_path": None,
        }
        self._cumulative_distance = 0.0

    def log_step(self, step_index: int, from_node: int, to_node: int,
                 segment_id: str, segment_name: Optional[str],
                 road_type: str, segment_length: float,
                 from_location: tuple[float, float],
                 to_location: tuple[float, float],
                 bearing: float, turn_angle: Optional[float],
                 busy_road_adjacent: bool,
                 phase: str,
                 chosen_score: float,
                 score_breakdown: dict,
                 alternatives: list[dict],
                 dist_to_start: float,
                 remaining_budget: float,
                 walk_buffers_count: int):
        """Record one step of the route planning."""
        self._cumulative_distance += segment_length
        step = {
            "step_index": step_index,
            "from_node": from_node,
            "to_node": to_node,
            "segment_id": segment_id,
            "segment_name": segment_name,
            "road_type": road_type,
            "segment_length_m": round(segment_length, 1),
            "cumulative_distance_m": round(self._cumulative_distance, 1),
            "from_location": {"lat": from_location[0], "lon": from_location[1]},
            "to_location": {"lat": to_location[0], "lon": to_location[1]},
            "bearing": round(bearing, 1),
            "turn_angle": round(turn_angle, 1) if turn_angle is not None else None,
            "busy_road_adjacent": busy_road_adjacent,
            "decision": {
                "phase": phase,
                "chosen_score": round(chosen_score, 2) if chosen_score != float("inf") else "inf",
                "score_breakdown": {k: round(v, 2) if isinstance(v, float) else v
                                    for k, v in score_breakdown.items()},
                "alternatives": alternatives,
            },
            "context": {
                "dist_to_start_m": round(dist_to_start, 1),
                "remaining_budget_m": round(remaining_budget, 1),
                "walk_buffers_count": walk_buffers_count,
            },
        }
        self.data["steps"].append(step)

    def log_return_path(self, trigger: str, triggered_at_step: int,
                        return_nodes: list[int], return_distance: float,
                        return_locations: Optional[list[tuple[float, float]]] = None):
        """Record the return path."""
        self.data["return_path"] = {
            "trigger": trigger,
            "triggered_at_step": triggered_at_step,
            "return_nodes": return_nodes,
            "return_distance_m": round(return_distance, 1),
        }
        if return_locations:
            self.data["return_path"]["return_locations"] = [
                {"lat": loc[0], "lon": loc[1]} for loc in return_locations
            ]

    def finalize(self, actual_distance: float):
        """Set final distance and return the complete data dict."""
        self.data["actual_distance_m"] = round(actual_distance, 1)
        return self.data

    def save(self, path: str):
        """Write route log to JSON file."""
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)


def evaluate_route(route_data: dict) -> dict:
    """Evaluate a route against quality criteria.

    Returns a dict with:
      - issues: list of {type, severity, description, steps}
      - stats: summary statistics
      - score: 0-100 overall quality score (higher is better)
    """
    steps = route_data.get("steps", [])
    rp = route_data.get("return_path") or {}
    target = route_data.get("target_distance_m", 0)
    actual = route_data.get("actual_distance_m", 0)
    start = route_data.get("start_location", {})

    issues = []

    # Pre-compute midpoints for spatial checks
    midpoints = []
    for s in steps:
        mid_lat = (s["from_location"]["lat"] + s["to_location"]["lat"]) / 2
        mid_lon = (s["from_location"]["lon"] + s["to_location"]["lon"]) / 2
        midpoints.append((mid_lat, mid_lon))

    # --- 1. Opposite-direction / parallel street detection ---
    # Only flag pairs that are far apart in route sequence (>= 10 steps)
    # but close in space — these indicate the route doubled back or used
    # both sides of a road. Consecutive segments are expected to be nearby.
    PROXIMITY_M = 60
    MIN_STEP_GAP = 10  # Must be this many steps apart in the route
    MIN_SEG_LEN = 15   # Skip very short intersection segments

    # Group issues by spatial cluster to avoid duplicates
    opposite_clusters: list[set[int]] = []
    parallel_clusters: list[set[int]] = []

    for i in range(len(steps)):
        if steps[i]["segment_length_m"] < MIN_SEG_LEN:
            continue
        for j in range(i + 1, len(steps)):
            if steps[j]["step_index"] - steps[i]["step_index"] < MIN_STEP_GAP:
                continue
            if steps[j]["segment_length_m"] < MIN_SEG_LEN:
                continue
            dist = haversine_distance(
                midpoints[i][0], midpoints[i][1],
                midpoints[j][0], midpoints[j][1],
            )
            if dist > PROXIMITY_M:
                continue

            bear_i = steps[i]["bearing"]
            bear_j = steps[j]["bearing"]
            diff = abs(((bear_i - bear_j + 180) % 360) - 180)

            if diff > 150:
                # Opposite directions — merge into cluster
                merged = False
                si, sj = steps[i]["step_index"], steps[j]["step_index"]
                for cluster in opposite_clusters:
                    if si in cluster or sj in cluster:
                        cluster.add(si)
                        cluster.add(sj)
                        merged = True
                        break
                if not merged:
                    opposite_clusters.append({si, sj})
            elif diff < 30:
                # Parallel — merge into cluster
                merged = False
                si, sj = steps[i]["step_index"], steps[j]["step_index"]
                for cluster in parallel_clusters:
                    if si in cluster or sj in cluster:
                        cluster.add(si)
                        cluster.add(sj)
                        merged = True
                        break
                if not merged:
                    parallel_clusters.append({si, sj})

    for cluster in opposite_clusters:
        sorted_steps = sorted(cluster)
        issues.append({
            "type": "opposite_direction",
            "severity": "high",
            "description": f"Steps {sorted_steps} walk opposite directions on nearby segments",
            "steps": sorted_steps,
        })

    for cluster in parallel_clusters:
        # Don't report parallel if already covered by opposite_direction
        opposite_steps = set()
        for c in opposite_clusters:
            opposite_steps.update(c)
        if cluster.issubset(opposite_steps):
            continue
        sorted_steps = sorted(cluster)
        issues.append({
            "type": "parallel_streets",
            "severity": "medium",
            "description": f"Steps {sorted_steps} use parallel nearby segments",
            "steps": sorted_steps,
        })

    # --- 3. Busy road usage ---
    busy_adj_steps = [s for s in steps if s.get("busy_road_adjacent")]
    busy_adj_dist = sum(s["segment_length_m"] for s in busy_adj_steps)
    busy_road_types = {"trunk", "primary", "secondary"}
    busy_walked = [s for s in steps if s["road_type"] in busy_road_types]
    crossing_steps = [s for s in steps
                      if s["decision"]["score_breakdown"].get("busy_road_penalty", 0) >= 30]

    for s in busy_walked:
        issues.append({
            "type": "busy_road_walked",
            "severity": "high",
            "description": f"Step {s['step_index']} walks on {s.get('segment_name') or s['road_type']} "
                           f"({s['road_type']}, {s['segment_length_m']:.0f}m)",
            "steps": [s["step_index"]],
        })

    if busy_adj_dist > actual * 0.15:
        issues.append({
            "type": "excessive_busy_adjacent",
            "severity": "medium",
            "description": f"{busy_adj_dist:.0f}m ({busy_adj_dist/actual*100:.0f}%) of route "
                           f"is adjacent to busy roads",
            "steps": [s["step_index"] for s in busy_adj_steps],
        })

    # --- 4. Return path quality ---
    return_dist = rp.get("return_distance_m", 0)
    return_pct = return_dist / actual * 100 if actual else 0
    if return_pct > 25:
        issues.append({
            "type": "long_return",
            "severity": "medium",
            "description": f"Return path is {return_dist:.0f}m ({return_pct:.0f}% of total route)",
            "steps": [],
        })

    # --- 5. Distance accuracy ---
    dist_error = abs(actual - target) / target * 100 if target else 0
    if dist_error > 20:
        issues.append({
            "type": "distance_error",
            "severity": "low",
            "description": f"Route is {actual:.0f}m vs target {target:.0f}m ({dist_error:.0f}% off)",
            "steps": [],
        })

    # --- 6. Loop shape ---
    max_dist_from_start = 0
    final_dist_from_start = 0
    if steps and start:
        for s in steps:
            d = s["context"]["dist_to_start_m"]
            max_dist_from_start = max(max_dist_from_start, d)
        last = steps[-1]
        final_dist_from_start = last["context"]["dist_to_start_m"]

    # Deduplicate: collapse parallel_streets that overlap with opposite_direction
    seen_pairs = set()
    deduped = []
    for issue in issues:
        if issue["type"] in ("opposite_direction", "parallel_streets"):
            pair = tuple(sorted(issue["steps"]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
        deduped.append(issue)
    issues = deduped

    # --- Compute score ---
    score = 100
    for issue in issues:
        if issue["severity"] == "high":
            score -= 10
        elif issue["severity"] == "medium":
            score -= 5
        elif issue["severity"] == "low":
            score -= 2
    score = max(0, score)

    # Stats
    explore_dist = actual - return_dist
    stats = {
        "total_distance_m": round(actual, 0),
        "target_distance_m": target,
        "explore_distance_m": round(explore_dist, 0),
        "return_distance_m": round(return_dist, 0),
        "return_pct": round(return_pct, 1),
        "total_steps": len(steps),
        "busy_adjacent_distance_m": round(busy_adj_dist, 0),
        "busy_adjacent_pct": round(busy_adj_dist / actual * 100, 1) if actual else 0,
        "busy_roads_walked": len(busy_walked),
        "crossing_penalty_steps": len(crossing_steps),
        "max_dist_from_start_m": round(max_dist_from_start, 0),
        "opposite_direction_issues": sum(1 for i in issues if i["type"] == "opposite_direction"),
        "parallel_street_issues": sum(1 for i in issues if i["type"] == "parallel_streets"),
    }

    return {"issues": issues, "stats": stats, "score": score}
