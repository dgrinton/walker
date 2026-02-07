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


def point_to_segment_distance(plat: float, plon: float,
                               alat: float, alon: float,
                               blat: float, blon: float) -> float:
    """Perpendicular distance from point P to line segment A-B in meters.

    Uses flat-earth approximation (suitable for short distances <1km).
    Projects P onto line A-B, clamped to the segment endpoints.
    """
    # Convert to flat-earth meters relative to A
    cos_lat = math.cos(math.radians(alat))
    ax, ay = 0.0, 0.0
    bx = (blon - alon) * cos_lat * 111_320
    by = (blat - alat) * 111_320
    px = (plon - alon) * cos_lat * 111_320
    py = (plat - alat) * 111_320

    # Vector AB
    abx = bx - ax
    aby = by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq < 1e-12:
        # Degenerate segment — return distance to A
        return math.hypot(px - ax, py - ay)

    # Parameter t of projection of P onto line AB, clamped to [0, 1]
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_sq))
    # Closest point on segment
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


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


def point_in_polygon(lat: float, lon: float, polygon: list[list[float]]) -> bool:
    """Check if a point is inside a polygon using ray-casting algorithm.

    Args:
        lat: Point latitude
        lon: Point longitude
        polygon: List of [lat, lon] pairs defining the polygon vertices
    """
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]

        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def _segments_intersect(ax1: float, ay1: float, ax2: float, ay2: float,
                        bx1: float, by1: float, bx2: float, by2: float) -> bool:
    """Check if two line segments (a1-a2) and (b1-b2) intersect using cross products."""
    def cross(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    d1 = cross(bx1, by1, bx2, by2, ax1, ay1)
    d2 = cross(bx1, by1, bx2, by2, ax2, ay2)
    d3 = cross(ax1, ay1, ax2, ay2, bx1, by1)
    d4 = cross(ax1, ay1, ax2, ay2, bx2, by2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


def segment_crosses_polygon(lat1: float, lon1: float, lat2: float, lon2: float,
                            polygon: list[list[float]]) -> bool:
    """Check if a line segment crosses into or through a polygon.

    Returns True if either endpoint is inside the polygon, or
    if the segment intersects any edge of the polygon.
    """
    if point_in_polygon(lat1, lon1, polygon) or point_in_polygon(lat2, lon2, polygon):
        return True

    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        py1, px1 = polygon[i]
        py2, px2 = polygon[j]
        if _segments_intersect(lat1, lon1, lat2, lon2, py1, px1, py2, px2):
            return True

    return False


def segment_crosses_any_polygon(lat1: float, lon1: float, lat2: float, lon2: float,
                                polygons: list[list[list[float]]]) -> bool:
    """Check if a line segment crosses into or through any of the given polygons."""
    return any(segment_crosses_polygon(lat1, lon1, lat2, lon2, poly) for poly in polygons)


def point_in_any_polygon(lat: float, lon: float, polygons: list[list[list[float]]]) -> bool:
    """Check if a point is inside any of the given polygons.

    Args:
        lat: Point latitude
        lon: Point longitude
        polygons: List of polygons, each a list of [lat, lon] pairs
    """
    return any(point_in_polygon(lat, lon, poly) for poly in polygons)


def segment_buffer_polygon(lat1: float, lon1: float, lat2: float, lon2: float,
                            width: float = 25, tip_angle: float = 60,
                            end_inset: float = 10,
                            min_length: float = 20) -> list[list[float]] | None:
    """Create a pointed hexagonal buffer polygon around a segment.

    The polygon has pointed tips inset from the segment endpoints so that
    intersections remain accessible.  For short segments the side points
    never reach full width, producing a 4-point diamond.

    Args:
        lat1, lon1: Start point of the segment
        lat2, lon2: End point of the segment
        width: Maximum width of the buffer in meters (default 25m)
        tip_angle: Angle at the pointed tips in degrees (default 60°)
        end_inset: Distance in meters to pull tips inward from endpoints (default 10m)
        min_length: Segments shorter than this produce no polygon (default 20m)

    Returns:
        List of [lat, lon] pairs defining the polygon vertices, or None if
        the segment is too short.
    """
    half_width = width / 2
    taper = half_width / math.tan(math.radians(tip_angle / 2))

    # Flat-earth conversion relative to midpoint
    mid_lat = (lat1 + lat2) / 2
    cos_lat = math.cos(math.radians(mid_lat))
    m_per_deg_lon = 111_320 * cos_lat
    m_per_deg_lat = 111_320

    # Segment vector in meters
    dx = (lon2 - lon1) * m_per_deg_lon
    dy = (lat2 - lat1) * m_per_deg_lat
    seg_len = math.hypot(dx, dy)

    if seg_len < min_length:
        return None

    # Unit vectors along and perpendicular to segment
    ux, uy = dx / seg_len, dy / seg_len  # along segment (A→B)
    px, py = -uy, ux  # perpendicular (left side when facing A→B)

    def offset_point(base_lat, base_lon, along_m, perp_m):
        """Offset a point by along_m along the segment and perp_m perpendicular."""
        total_dx = along_m * ux + perp_m * px
        total_dy = along_m * uy + perp_m * py
        return [base_lat + total_dy / m_per_deg_lat,
                base_lon + total_dx / m_per_deg_lon]

    # Tip positions are inset from the endpoints
    tip_a = offset_point(lat1, lon1, end_inset, 0)
    tip_b = offset_point(lat2, lon2, -end_inset, 0)

    # Effective length between the two tips
    inner_len = seg_len - 2 * end_inset
    if inner_len < 1e-6:
        return None

    if inner_len >= 2 * taper:
        # 6-point hexagon
        return [
            tip_a,                                                 # Tip A
            offset_point(lat1, lon1, end_inset + taper, half_width),   # Left side near A
            offset_point(lat2, lon2, -(end_inset + taper), half_width),  # Left side near B
            tip_b,                                                 # Tip B
            offset_point(lat2, lon2, -(end_inset + taper), -half_width),  # Right side near B
            offset_point(lat1, lon1, end_inset + taper, -half_width),  # Right side near A
        ]
    else:
        # 4-point diamond: inner length too short for full width
        mid_width = (inner_len / 2) * math.tan(math.radians(tip_angle / 2))
        mid_lon = (lon1 + lon2) / 2
        return [
            tip_a,                                                 # Tip A
            offset_point(mid_lat, mid_lon, 0, mid_width),          # Left midpoint
            tip_b,                                                 # Tip B
            offset_point(mid_lat, mid_lon, 0, -mid_width),         # Right midpoint
        ]


def are_bearings_parallel(bearing1: float, bearing2: float, threshold: float = 30) -> bool:
    """Check if two bearings are parallel (same OR opposite direction).

    Normalizes both bearings to 0-180 range so that e.g. 45° and 225° are
    considered parallel (both map to 45°).
    """
    b1 = bearing1 % 360
    b2 = bearing2 % 360
    # Collapse to 0-180 range (direction-agnostic)
    if b1 >= 180:
        b1 -= 180
    if b2 >= 180:
        b2 -= 180
    diff = abs(b1 - b2)
    # Handle wrap-around at 0/180 boundary
    if diff > 90:
        diff = 180 - diff
    return diff < threshold


def is_parallel_corridor(
    graph: "StreetGraph",
    candidate_from: int,
    candidate_to: int,
    recent_seg: "DirectedSegment",
    angle_threshold: float = 30,
    distance_threshold: float = 50,
) -> bool:
    """Check if a candidate segment runs parallel to a recent segment.

    Returns True if:
    - The bearings are parallel (same OR opposite direction, within angle_threshold)
    - The segments are close (midpoint of each within distance_threshold of the other's line)

    This catches:
    - Same street name, opposite direction (definite backtracking)
    - Parallel sidewalks/paths in either direction (corridor duplication)
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

    # Check if bearings are parallel (same or opposite direction)
    if not are_bearings_parallel(cand_bearing, recent_bearing, angle_threshold):
        return False

    # Calculate midpoints
    cand_mid_lat = (cand_from_loc[0] + cand_to_loc[0]) / 2
    cand_mid_lon = (cand_from_loc[1] + cand_to_loc[1]) / 2
    recent_mid_lat = (recent_from_loc[0] + recent_to_loc[0]) / 2
    recent_mid_lon = (recent_from_loc[1] + recent_to_loc[1]) / 2

    # Check proximity: midpoint of each segment to the other's line
    dist_cand_mid_to_recent = point_to_segment_distance(
        cand_mid_lat, cand_mid_lon,
        recent_from_loc[0], recent_from_loc[1],
        recent_to_loc[0], recent_to_loc[1],
    )
    dist_recent_mid_to_cand = point_to_segment_distance(
        recent_mid_lat, recent_mid_lon,
        cand_from_loc[0], cand_from_loc[1],
        cand_to_loc[0], cand_to_loc[1],
    )

    return min(dist_cand_mid_to_recent, dist_recent_mid_to_cand) < distance_threshold
