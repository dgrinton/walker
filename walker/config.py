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
        "residential": 100,
        "living_street": 100,
        "service": 100,
        "unclassified": 100,
        "tertiary": 100,
        "secondary": 100,
        "primary": 100,
        "trunk": 100,
        "motorway": 100,  # effectively blocked
        "virtual": 2,     # virtual edges (close-node shortcuts) — slightly above footway
    },
    "default_road_weight": 5,
    # Walk buffer polygon (anti-backtracking)
    "walk_buffer_width": 50,       # meters - max width of hexagonal buffer around walked segments
    "walk_buffer_tip_angle": 60,   # degrees - angle at the pointed tips of the buffer polygon
    "walk_buffer_end_inset": 10,   # meters - pull tips inward from segment endpoints
    "walk_buffer_min_length": 20,  # meters - skip buffer creation for segments shorter than this
    # Busy road proximity penalty (for footpaths alongside busy roads)
    "busy_road_types": {"secondary", "primary", "trunk"},
    "footpath_types": {"footway", "path", "pedestrian"},
    "busy_road_proximity_threshold": 30,  # meters
    "busy_road_proximity_penalty": 8,     # added to score (makes footway=1 effectively score like primary=9)
    "distance_milestone_interval": 250,  # meters between distance announcements
    "dead_end_lookahead": 10,  # steps to look ahead for dead-end detection
    "dead_end_penalty": 40,  # penalty for edges leading into dead ends
    # Virtual edges (close-node shortcuts)
    "virtual_edge_max_distance": 15,  # meters — max gap to bridge with virtual edge
    # Convexity bias (loop shaping)
    "convexity_onset": 0.35,   # fraction of target distance before bias kicks in
    "convexity_weight": 0.5,   # penalty per meter of delta (moving away from start)
}
