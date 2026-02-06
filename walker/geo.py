"""Geographic utility functions."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import DirectedSegment
    from .graph import StreetGraph


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters using Haversine formula"""
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
        The result of func() on success, or None if all retries failed
    """
    start_time = time.time()
    delay = initial_delay
    attempt = 1

    while True:
        result = func()
        if result:
            return result

        elapsed = time.time() - start_time
        if elapsed >= max_time:
            print(f"Failed to complete {description} after {elapsed:.1f}s ({attempt} attempts)")
            return None

        remaining = max_time - elapsed
        sleep_time = min(delay, remaining, max_delay)
        if sleep_time > 0:
            print(f"Retrying {description} in {sleep_time:.1f}s (attempt {attempt})...")
            time.sleep(sleep_time)

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


def is_opposite_direction(
    graph: "StreetGraph",
    candidate_from: int,
    candidate_to: int,
    recent_seg: "DirectedSegment",
    angle_threshold: float = 30,
    distance_threshold: float = 50,
) -> bool:
    """Check if a candidate segment is going the opposite direction of a recent segment.

    Returns True only if:
    - The candidate segment's bearing is ~180° from the recent segment's bearing
      (within angle_threshold of being anti-parallel)
    - The midpoints of the two segments are close (within distance_threshold meters)

    This catches both:
    - Same street name, opposite direction (definite backtracking)
    - Parallel sidewalks/paths going opposite ways (likely backtracking)

    Does NOT return True for:
    - Continuing in the same direction (same or similar bearing)
    - Perpendicular streets (crossing, not backtracking)
    """
    # Get node locations
    cand_from_loc = graph.get_node_location(candidate_from)
    cand_to_loc = graph.get_node_location(candidate_to)
    recent_from_loc = graph.get_node_location(recent_seg.from_node)
    recent_to_loc = graph.get_node_location(recent_seg.to_node)

    if not all([cand_from_loc, cand_to_loc, recent_from_loc, recent_to_loc]):
        return False

    # Calculate bearings
    cand_bearing = bearing_between(
        cand_from_loc[0], cand_from_loc[1], cand_to_loc[0], cand_to_loc[1]
    )
    recent_bearing = bearing_between(
        recent_from_loc[0], recent_from_loc[1], recent_to_loc[0], recent_to_loc[1]
    )

    # Check if bearings are anti-parallel (approximately 180° apart)
    bearing_diff = abs((cand_bearing - recent_bearing + 180) % 360 - 180)
    # bearing_diff is now 0 if same direction, 180 if opposite
    # We want to check if it's close to 180 (opposite)
    is_anti_parallel = abs(bearing_diff - 180) < angle_threshold

    if not is_anti_parallel:
        return False

    # Calculate midpoints
    cand_mid_lat = (cand_from_loc[0] + cand_to_loc[0]) / 2
    cand_mid_lon = (cand_from_loc[1] + cand_to_loc[1]) / 2
    recent_mid_lat = (recent_from_loc[0] + recent_to_loc[0]) / 2
    recent_mid_lon = (recent_from_loc[1] + recent_to_loc[1]) / 2

    # Check if midpoints are close
    midpoint_distance = haversine_distance(
        cand_mid_lat, cand_mid_lon, recent_mid_lat, recent_mid_lon
    )

    return midpoint_distance < distance_threshold
