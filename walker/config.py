"""Configuration settings for Walker."""

CONFIG = {
    "gps_poll_interval": 3,  # seconds
    "intersection_arrival_radius": 20,  # meters
    "direction_warning_distance": 20,  # meters
    "route_deviation_threshold": 50,  # meters - recalculate if user deviates this far
    "default_walk_distance": 2000,  # meters (2km default)
    "osm_fetch_radius": 1500,  # meters - area to fetch from OSM
    "log_interval": 10,  # seconds between log entries
    "direction_speak_interval": 30,  # seconds between periodic direction announcements
    "min_waypoint_distance": 20,  # meters - skip waypoints closer than this
    # Road type weights (lower = preferred)
    "road_weights": {
        "footway": 1,
        "pedestrian": 1,
        "path": 1,
        "residential": 2,
        "living_street": 2,
        "service": 5,
        "unclassified": 4,
        "tertiary": 5,
        "secondary": 7,
        "primary": 20,
        "trunk": 50,
        "motorway": 100,  # effectively blocked
        "virtual": 2,     # virtual edges (close-node shortcuts) — slightly above footway
    },
    "default_road_weight": 5,
    # Walk buffer polygon (anti-backtracking)
    "walk_buffer_width": 30,       # meters - max width of hexagonal buffer around walked segments
    "walk_buffer_tip_angle": 60,   # degrees - angle at the pointed tips of the buffer polygon
    "walk_buffer_end_inset": 10,   # meters - pull tips inward from segment endpoints
    "walk_buffer_min_length": 10,  # meters - skip buffer creation for segments shorter than this
    # Busy road proximity / crossing penalties
    "busy_road_types": {"secondary", "primary", "trunk"},
    "busy_road_proximity_threshold": 50,  # meters
    "busy_road_proximity_penalty": 15,    # added to score for segments near busy roads
    "busy_road_crossing_penalty": 30,     # added to score for segments sharing a node with a busy road
    "distance_milestone_interval": 250,  # meters between distance announcements
    "dead_end_lookahead": 10,  # steps to look ahead for dead-end detection
    "dead_end_penalty": 40,  # penalty for edges leading into dead ends
    # Virtual edges (close-node shortcuts)
    "virtual_edge_max_distance": 0,   # meters — 0 disables virtual edges
    # Loop steering — constant curvature to form a circuit
    "loop_steering_weight": 10,            # max penalty for worst-direction edge (180° off ideal)
    "loop_steering_dist_weight": 0.03,     # penalty per meter beyond ideal loop radius
    "visited_node_penalty": 15, # penalty per previous visit to target node (trap escape)
    "homeward_bonus_weight": 0.15,  # per-meter bonus for moving closer to start (scales with progress²)
    "parallel_segment_penalty": 20,  # penalty for edges running parallel/opposite to walked segments
    "parallel_segment_proximity": 25,  # meters — max distance for parallel-segment detection
    # Intersection simplification
    "simplify_max_segment": 20,  # meters — max individual segment length to merge at degree-2 nodes (same-way only)
}
