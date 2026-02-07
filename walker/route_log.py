"""Route logging for iterative route improvement."""

import json
from datetime import datetime, timezone
from typing import Optional

from .config import CONFIG


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
                        return_nodes: list[int], return_distance: float):
        """Record the return path."""
        self.data["return_path"] = {
            "trigger": trigger,
            "triggered_at_step": triggered_at_step,
            "return_nodes": return_nodes,
            "return_distance_m": round(return_distance, 1),
        }

    def finalize(self, actual_distance: float):
        """Set final distance and return the complete data dict."""
        self.data["actual_distance_m"] = round(actual_distance, 1)
        return self.data

    def save(self, path: str):
        """Write route log to JSON file."""
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
