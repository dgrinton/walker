"""Configuration settings for Walker."""

CONFIG = {
    "gps_poll_interval": 3,  # seconds
    "intersection_arrival_radius": 20,  # meters
    "direction_warning_distance": 20,  # meters
    "route_deviation_threshold": 50,  # meters - recalculate if user deviates this far
    "default_walk_distance": 2000,  # meters (2km default)
    "osm_fetch_radius": 1500,  # meters - area to fetch from OSM
    "log_interval": 10,  # seconds between log entries
    "min_waypoint_distance": 20,  # meters - skip waypoints closer than this
    # Road type weights (lower = preferred)
    "road_weights": {
        "footway": 1,
        "pedestrian": 1,
        "path": 1,
        "residential": 2,
        "living_street": 2,
        "service": 3,
        "unclassified": 4,
        "tertiary": 5,
        "secondary": 7,
        "primary": 9,
        "trunk": 15,
        "motorway": 100,  # effectively blocked
    },
    "default_road_weight": 5,
    # Backtracking penalties (to avoid zigzag on parallel streets)
    "same_street_penalty": 25,
    "parallel_segment_penalty": 30,
    "parallel_angle_threshold": 30,    # degrees - how close to 180° counts as opposite
    "parallel_distance_threshold": 50,  # meters - max distance between parallel segments
    "recent_segment_history": 10,  # number of recent segments to check for backtracking
    # Busy road proximity penalty (for footpaths alongside busy roads)
    "busy_road_types": {"secondary", "primary", "trunk"},
    "footpath_types": {"footway", "path", "pedestrian"},
    "busy_road_proximity_threshold": 30,  # meters
    "busy_road_proximity_penalty": 8,     # added to score (makes footway=1 effectively score like primary=9)
    "distance_milestone_interval": 250,  # meters between distance announcements
    "corridor_min_segment_length": 10,  # meters - skip short segments in corridor detection
    "corridor_name_proximity": 200,  # meters - midpoint distance for same-name corridor grouping
    "dead_end_lookahead": 10,  # steps to look ahead for dead-end detection
    "dead_end_penalty": 40,  # penalty for edges leading into dead ends
}
