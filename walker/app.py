"""Main Walker application."""

import json
import time
from datetime import datetime
from typing import Optional

from .config import CONFIG
from .models import Location
from .logger import Logger
from .gps import GPS, GPSRecorder, GPSPlayback
from .debug_gui import DebugServer, WebSocketGPS
from .geo import haversine_distance, bearing_between, bearing_to_compass, retry_with_backoff, segment_buffer_polygon
from .osm import OSMFetcher
from .graph import StreetGraph
from .history import HistoryDB
from .audio import Audio
from .planner import RoutePlanner


class Walker:
    """Main application"""

    def __init__(self, log_path: Optional[str] = None,
                 preview_mode: bool = False, execute_mode: bool = False,
                 start_location: Optional[tuple[float, float]] = None,
                 html_output: Optional[str] = None,
                 gpx_output: Optional[str] = None,
                 debug_gui: bool = False):
        self.gps = GPS()
        self.audio = Audio()
        self.history = HistoryDB()
        self.graph: Optional[StreetGraph] = None
        self.planner: Optional[RoutePlanner] = None
        self.preview_mode = preview_mode
        self.execute_mode = execute_mode
        self.start_location = start_location  # (lat, lon) tuple for testing
        self.html_output = html_output  # HTML file for route visualization
        self.gpx_output = gpx_output  # GPX file for navigation apps
        self.debug_gui = debug_gui

        # Debug GUI server
        self.debug_server: Optional[DebugServer] = None
        if debug_gui:
            self.debug_server = DebugServer()
            self.debug_server.start()
            # Set up audio callback
            Audio.set_callback(self.debug_server.send_audio)

        # Logger with optional callback for debug GUI
        log_callback = self.debug_server.send_log if self.debug_server else None
        self.logger = Logger(log_path, callback=log_callback)

        self.current_location: Optional[Location] = None
        self.current_node: Optional[int] = None
        self.previous_node: Optional[int] = None
        self.next_node: Optional[int] = None

        self.walk_id: Optional[int] = None
        self.segments_walked: int = 0
        self._last_counted_index: int = 0  # Last route index for which distance was counted

        # Verbose mode timing
        self.last_distance_milestone = 0
        self.last_log_update = 0
        self.walk_start_time = 0

        # GPS source (can be swapped for recording/playback)
        self.gps_source = self.gps

    def set_gps_source(self, source):
        """Set GPS source (GPS, GPSRecorder, or GPSPlayback)"""
        self.gps_source = source

    def get_state(self) -> dict:
        """Get current state as dict for logging"""
        state = {
            "walked_distance": self.planner.walked_distance if self.planner else 0,
            "target_distance": self.planner.target_distance if self.planner else 0,
            "segments_walked": self.segments_walked,
            "current_node": self.current_node,
            "next_node": self.next_node,
            "gps_status": self.gps_source.get_status() if hasattr(self.gps_source, 'get_status') else "unknown",
        }
        if self.current_location:
            state["location"] = {
                "lat": self.current_location.lat,
                "lon": self.current_location.lon,
                "accuracy": self.current_location.accuracy
            }
        # Add next node location for debug GUI
        if self.next_node and self.graph:
            next_loc = self.graph.get_node_location(self.next_node)
            if next_loc:
                state["next_node_location"] = {"lat": next_loc[0], "lon": next_loc[1]}
        # Add route index for debug GUI
        if self.planner:
            state["route_index"] = self.planner.route_index
        return state

    def _send_route_to_debug_server(self, start_location: Location):
        """Send route data to debug server for visualization"""
        if not self.debug_server or not self.planner or not self.graph:
            return

        route = self.planner.planned_route

        # Build route coordinates
        route_coords = []
        for node_id in route:
            loc = self.graph.get_node_location(node_id)
            if loc:
                route_coords.append([loc[0], loc[1]])

        # Build segment data with novelty info
        segments = []
        for i in range(len(route) - 1):
            node1, node2 = route[i], route[i + 1]
            loc1 = self.graph.get_node_location(node1)
            loc2 = self.graph.get_node_location(node2)
            segment = self.graph.get_segment(node1, node2)

            if loc1 and loc2 and segment:
                times_walked, _ = self.history.get_segment_history(segment.id)
                buffer = segment_buffer_polygon(
                    loc1[0], loc1[1], loc2[0], loc2[1],
                    width=CONFIG["walk_buffer_width"],
                    tip_angle=CONFIG["walk_buffer_tip_angle"],
                    end_inset=CONFIG["walk_buffer_end_inset"],
                    min_length=CONFIG["walk_buffer_min_length"],
                )
                segments.append({
                    "coords": [[loc1[0], loc1[1]], [loc2[0], loc2[1]]],
                    "name": segment.name,
                    "is_new": times_walked == 0,
                    "times_walked": times_walked,
                    "length": segment.length,
                    "buffer": buffer,
                })

        self.debug_server.send_route({
            "route": route_coords,
            "segments": segments,
            "start": {"lat": start_location.lat, "lon": start_location.lon}
        })

    def periodic_update(self):
        """Handle periodic status updates"""
        now = time.time()

        # Log to file every 10 seconds
        if now - self.last_log_update >= CONFIG["log_interval"]:
            self.logger.log("STATE", self.get_state())
            self.last_log_update = now

    def initialize(self, target_distance: float) -> bool:
        """Initialize walk with GPS fix and map data"""

        self.logger.log("Initializing walk", {"target_distance": target_distance})

        # Use provided start location if available (for testing)
        if self.start_location:
            lat, lon = self.start_location
            location = Location(lat=lat, lon=lon, accuracy=0, timestamp=time.time())
            self.logger.log("Using provided start location", {"lat": lat, "lon": lon})
            print(f"Using provided location: {lat:.5f}, {lon:.5f}")
        else:
            print("Getting GPS fix...")
            if not self.preview_mode and not self.execute_mode:
                self.audio.speak("Getting GPS fix")

            # Get GPS fix with retry and backoff (up to 30s)
            def try_gps():
                loc = self.gps_source.get_location(timeout=10)
                if loc:
                    self.logger.log("GPS fix obtained", {"lat": loc.lat, "lon": loc.lon})
                else:
                    self.logger.log("GPS attempt failed")
                return loc

            location = retry_with_backoff(
                try_gps,
                max_time=30.0,
                initial_delay=1.0,
                max_delay=8.0,
                description="GPS fix"
            )

            if not location:
                self.logger.log("Could not get GPS location after retries")
                print("Could not get GPS location")
                self.audio.speak("Could not get GPS location")
                return False

            self.logger.log("Got GPS fix", {"lat": location.lat, "lon": location.lon, "accuracy": location.accuracy})
            print(f"Location: {location.lat:.5f}, {location.lon:.5f} (accuracy: {location.accuracy}m)")

        self.current_location = location

        # Fetch OSM data with retry and backoff (up to 30s)
        osm_data = None

        def try_osm():
            data = OSMFetcher.fetch_streets(
                location.lat, location.lon, CONFIG["osm_fetch_radius"]
            )
            if data.get("elements"):
                self.logger.log("OSM data fetched", {"elements": len(data["elements"])})
                return data
            self.logger.log("OSM fetch returned no elements")
            return None

        osm_data = retry_with_backoff(
            try_osm,
            max_time=30.0,
            initial_delay=2.0,
            max_delay=8.0,
            description="Map data fetch"
        )

        if not osm_data:
            self.logger.log("Could not fetch map data after retries")
            print("Could not fetch map data")
            self.audio.speak("Could not fetch map data")
            return False

        # Build graph
        self.graph = StreetGraph()
        self.graph.build_from_osm(osm_data)
        self.logger.log("Built graph", {"nodes": len(self.graph.nodes), "segments": len(self.graph.segments)})

        # Find starting node
        self.current_node = self.graph.find_nearest_node(location.lat, location.lon)
        if not self.current_node:
            self.logger.log("Could not find starting point on map")
            print("Could not find starting point on map")
            return False

        node_loc = self.graph.get_node_location(self.current_node)
        self.logger.log("Starting node", {"node": self.current_node, "lat": node_loc[0], "lon": node_loc[1]})
        print(f"Starting at node {self.current_node} ({node_loc[0]:.5f}, {node_loc[1]:.5f})")

        # Initialize planner with exclusion zones
        zones = self.history.get_exclusion_zones()
        excluded_polygons = [z['polygon'] for z in zones] if zones else None
        self.planner = RoutePlanner(self.graph, self.history, excluded_zones=excluded_polygons)

        # Calculate the full route upfront
        print("Calculating route...")
        route = self.planner.calculate_full_route(self.current_node, target_distance)
        route_distance = self.planner.get_route_distance()
        self.logger.log("Route calculated", {
            "nodes": len(route),
            "distance": route_distance,
            "target": target_distance
        })
        print(f"Route calculated: {len(route)} nodes, {route_distance:.0f}m")

        # Send route to debug server if active
        if self.debug_server:
            self._send_route_to_debug_server(location)

        # Preview mode: display route and optionally execute
        if self.preview_mode:
            self.display_route_preview()
            # If also executing, continue to execute_route
            if self.execute_mode:
                return True
            return False

        # Execute mode: run through route without GPS
        if self.execute_mode:
            return True

        # Start walk in database
        self.walk_id = self.history.start_walk()

        # Get first waypoint (skipping any too close to start)
        self.next_node = self._get_next_waypoint()

        self.walk_start_time = time.time()
        self.last_log_update = time.time()

        # Immediate audio update with directions to first waypoint
        if self.next_node:
            self._speak_status()
            self.logger.log("Walk started", {"next_node": self.next_node})
            print(f"Walk started, heading to node {self.next_node}")

        return True

    def display_route_preview(self):
        """Display a preview of the calculated route"""
        if not self.planner or not self.planner.planned_route:
            print("No route to preview")
            return

        print("\n" + "=" * 60)
        print("ROUTE PREVIEW")
        print("=" * 60)

        segments = self.planner.get_route_segments()
        total_distance = self.planner.get_route_distance()

        print(f"\nTotal distance: {total_distance:.0f}m ({total_distance/1000:.2f}km)")
        print(f"Total segments: {len(segments)}")
        print(f"Total nodes: {len(self.planner.planned_route)}")

        # Count new vs walked segments
        new_segments = 0
        walked_segments = 0
        for seg in segments:
            times_walked, _ = self.history.get_segment_history(seg.id)
            if times_walked == 0:
                new_segments += 1
            else:
                walked_segments += 1

        print(f"\nNew segments: {new_segments}")
        print(f"Previously walked: {walked_segments}")
        if segments:
            print(f"Novelty: {new_segments / len(segments) * 100:.1f}%")

        print("\n" + "-" * 60)
        print("TURN-BY-TURN DIRECTIONS")
        print("-" * 60)

        route = self.planner.planned_route
        cumulative_distance = 0

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            segment = self.graph.get_segment(current_node, next_node)

            if not segment:
                continue

            # Check if this is an intersection (turn point)
            is_intersection = self.graph.is_intersection(next_node)

            # Get direction for this move
            current_loc = self.graph.get_node_location(current_node)
            next_loc = self.graph.get_node_location(next_node)

            if current_loc and next_loc:
                bearing = bearing_between(
                    current_loc[0], current_loc[1], next_loc[0], next_loc[1]
                )
                compass = bearing_to_compass(bearing)

                # Get relative direction if we have a previous node
                if i > 0:
                    prev_node = route[i - 1]
                    instruction = self.planner.get_direction_instruction(
                        prev_node, current_node, next_node
                    )
                else:
                    instruction = f"Head {compass}"

                # Only show at intersections or significant turns
                if i == 0 or is_intersection:
                    street_name = segment.name or "unnamed"
                    times_walked, _ = self.history.get_segment_history(segment.id)
                    novelty = " [NEW]" if times_walked == 0 else f" [x{times_walked}]"

                    print(f"\n{cumulative_distance:>6.0f}m | {instruction}")
                    print(f"         -> {street_name}{novelty} ({segment.length:.0f}m)")

            cumulative_distance += segment.length

        print(f"\n{cumulative_distance:>6.0f}m | Arrive at start")
        print("\n" + "=" * 60)

        # Generate HTML visualization if requested
        if self.html_output:
            self.generate_route_html()

        # Generate GPX file if requested
        if self.gpx_output:
            self.generate_route_gpx()

    def generate_route_html(self):
        """Generate an HTML file with a map visualization of the route"""
        if not self.planner or not self.planner.planned_route:
            print("No route to visualize")
            return

        route = self.planner.planned_route
        segments = self.planner.get_route_segments()

        # Collect coordinates for the route
        coordinates = []
        for node_id in route:
            loc = self.graph.get_node_location(node_id)
            if loc:
                coordinates.append(loc)

        if not coordinates:
            print("No coordinates to visualize")
            return

        # Calculate map center and bounds
        lats = [c[0] for c in coordinates]
        lons = [c[1] for c in coordinates]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        # Build segment data with colors based on novelty
        segment_data = []
        for i in range(len(route) - 1):
            node1, node2 = route[i], route[i + 1]
            loc1 = self.graph.get_node_location(node1)
            loc2 = self.graph.get_node_location(node2)
            segment = self.graph.get_segment(node1, node2)

            if loc1 and loc2 and segment:
                times_walked, _ = self.history.get_segment_history(segment.id)
                is_new = times_walked == 0
                street_name = segment.name or "unnamed"
                segment_data.append({
                    "coords": [[loc1[0], loc1[1]], [loc2[0], loc2[1]]],
                    "name": street_name,
                    "new": is_new,
                    "times_walked": times_walked,
                    "length": segment.length
                })

        # Build turn markers at intersections
        turn_markers = []
        for i, node_id in enumerate(route):
            if self.graph.is_intersection(node_id) or i == 0:
                loc = self.graph.get_node_location(node_id)
                if loc:
                    label = "Start" if i == 0 else f"Turn {len(turn_markers)}"
                    turn_markers.append({
                        "lat": loc[0],
                        "lon": loc[1],
                        "label": label
                    })

        # Generate HTML with Leaflet.js
        segments_json = json.dumps(segment_data)
        markers_json = json.dumps(turn_markers)
        total_distance = self.planner.get_route_distance()
        new_count = sum(1 for s in segment_data if s["new"])
        walked_count = len(segment_data) - new_count

        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Walker Route Preview</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        #map {{ position: absolute; top: 0; bottom: 0; left: 0; right: 300px; }}
        #sidebar {{ position: absolute; top: 0; bottom: 0; right: 0; width: 300px; background: #f5f5f5; overflow-y: auto; padding: 15px; box-sizing: border-box; }}
        h2 {{ margin-top: 0; }}
        .stat {{ margin: 10px 0; padding: 10px; background: white; border-radius: 5px; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .stat-value {{ font-size: 24px; font-weight: bold; }}
        .legend {{ margin-top: 20px; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .legend-color {{ width: 30px; height: 4px; margin-right: 10px; }}
        .new {{ background: #22c55e; }}
        .walked {{ background: #3b82f6; }}
        .segment-list {{ margin-top: 20px; max-height: 300px; overflow-y: auto; }}
        .segment-item {{ padding: 8px; background: white; margin: 5px 0; border-radius: 3px; font-size: 13px; border-left: 4px solid #ccc; }}
        .segment-item.new {{ border-left-color: #22c55e; }}
        .segment-item.walked {{ border-left-color: #3b82f6; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="sidebar">
        <h2>Route Preview</h2>
        <div class="stat">
            <div class="stat-label">Total Distance</div>
            <div class="stat-value">{total_distance/1000:.2f} km</div>
        </div>
        <div class="stat">
            <div class="stat-label">Segments</div>
            <div class="stat-value">{len(segment_data)}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Novelty</div>
            <div class="stat-value">{new_count / len(segment_data) * 100 if segment_data else 0:.0f}%</div>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-color new"></div> New segments ({new_count})</div>
            <div class="legend-item"><div class="legend-color walked"></div> Previously walked ({walked_count})</div>
        </div>
        <div class="segment-list">
            <h3>Segments</h3>
            <div id="segments"></div>
        </div>
    </div>
    <script>
        var segments = {segments_json};
        var markers = {markers_json};

        var map = L.map('map').setView([{center_lat}, {center_lon}], 15);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        // Draw route segments
        var bounds = [];
        segments.forEach(function(seg, idx) {{
            var color = seg.new ? '#22c55e' : '#3b82f6';
            var line = L.polyline(seg.coords, {{
                color: color,
                weight: 5,
                opacity: 0.8
            }}).addTo(map);
            line.bindPopup('<b>' + seg.name + '</b><br>' +
                          seg.length.toFixed(0) + 'm' +
                          (seg.new ? '<br><em>New!</em>' : '<br>Walked ' + seg.times_walked + 'x'));
            bounds.push(seg.coords[0]);
            bounds.push(seg.coords[1]);
        }});

        // Add turn markers
        markers.forEach(function(m, idx) {{
            var marker = L.circleMarker([m.lat, m.lon], {{
                radius: idx === 0 ? 10 : 6,
                fillColor: idx === 0 ? '#ef4444' : '#ffffff',
                color: '#333',
                weight: 2,
                fillOpacity: 1
            }}).addTo(map);
            marker.bindPopup(m.label);
        }});

        // Fit map to route bounds
        if (bounds.length > 0) {{
            map.fitBounds(bounds, {{ padding: [20, 20] }});
        }}

        // Build segment list
        var segList = document.getElementById('segments');
        segments.forEach(function(seg, idx) {{
            var div = document.createElement('div');
            div.className = 'segment-item ' + (seg.new ? 'new' : 'walked');
            div.innerHTML = '<strong>' + seg.name + '</strong><br>' +
                           seg.length.toFixed(0) + 'm' +
                           (seg.new ? ' - New' : ' - Walked ' + seg.times_walked + 'x');
            segList.appendChild(div);
        }});
    </script>
</body>
</html>'''

        # Write HTML file
        with open(self.html_output, 'w') as f:
            f.write(html)

        print(f"\nRoute visualization saved to: {self.html_output}")

    def generate_route_gpx(self):
        """Generate a GPX file for use in OsmAnd or other GPS navigation apps"""
        if not self.planner or not self.planner.planned_route:
            print("No route to export")
            return

        route = self.planner.planned_route
        total_distance = self.planner.get_route_distance()
        timestamp = datetime.now().isoformat()

        # Build waypoints at intersections with turn instructions
        waypoints = []
        for i, node_id in enumerate(route):
            loc = self.graph.get_node_location(node_id)
            if not loc:
                continue

            # Only add waypoints at start, intersections, and end
            is_start = (i == 0)
            is_end = (i == len(route) - 1)
            is_intersection = self.graph.is_intersection(node_id)

            if is_start or is_end or is_intersection:
                if is_start:
                    name = "Start"
                elif is_end:
                    name = "End"
                else:
                    # Get turn instruction for this intersection
                    if i > 0 and i < len(route) - 1:
                        prev_node = route[i - 1]
                        next_node = route[i + 1]
                        instruction = self.planner.get_direction_instruction(
                            prev_node, node_id, next_node
                        )
                        name = instruction.capitalize()
                    else:
                        name = f"Waypoint {len(waypoints) + 1}"

                waypoints.append({
                    "lat": loc[0],
                    "lon": loc[1],
                    "name": name
                })

        # Build track points for the route line
        trackpoints = []
        for node_id in route:
            loc = self.graph.get_node_location(node_id)
            if loc:
                trackpoints.append({"lat": loc[0], "lon": loc[1]})

        # Generate GPX XML
        gpx_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="Walker"',
            '     xmlns="http://www.topografix.com/GPX/1/1"',
            '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '     xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">',
            '  <metadata>',
            f'    <name>Walker Route ({total_distance/1000:.2f} km)</name>',
            f'    <time>{timestamp}</time>',
            '  </metadata>',
        ]

        # Add waypoints
        for wp in waypoints:
            gpx_lines.append(f'  <wpt lat="{wp["lat"]:.6f}" lon="{wp["lon"]:.6f}">')
            gpx_lines.append(f'    <name>{wp["name"]}</name>')
            gpx_lines.append('  </wpt>')

        # Add track
        gpx_lines.append('  <trk>')
        gpx_lines.append(f'    <name>Walker Route</name>')
        gpx_lines.append('    <trkseg>')
        for tp in trackpoints:
            gpx_lines.append(f'      <trkpt lat="{tp["lat"]:.6f}" lon="{tp["lon"]:.6f}"/>')
        gpx_lines.append('    </trkseg>')
        gpx_lines.append('  </trk>')
        gpx_lines.append('</gpx>')

        # Write GPX file
        gpx_content = '\n'.join(gpx_lines)
        with open(self.gpx_output, 'w') as f:
            f.write(gpx_content)

        print(f"\nGPX route saved to: {self.gpx_output}")
        print(f"  {len(waypoints)} waypoints, {len(trackpoints)} track points")
        print("  Import into OsmAnd: Menu -> My Places -> Tracks -> Import")

    def execute_route(self):
        """Execute the planned route virtually and add to database"""
        if not self.planner or not self.planner.planned_route:
            print("No route to execute")
            return

        print("\n" + "=" * 60)
        print("EXECUTING ROUTE (Virtual Walk)")
        print("=" * 60)

        # Start walk in database
        self.walk_id = self.history.start_walk()

        route = self.planner.planned_route
        total_distance = 0

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            segment = self.graph.get_segment(current_node, next_node)

            if segment:
                # Record this segment as walked
                self.history.record_segment(segment.id)
                total_distance += segment.length
                self.segments_walked += 1

                street_name = segment.name or "unnamed"
                print(f"  Walked: {street_name} ({segment.length:.0f}m) - Total: {total_distance:.0f}m")

        # End walk in database
        self.history.end_walk(self.walk_id, total_distance, self.segments_walked)

        print("\n" + "-" * 60)
        print("EXECUTION COMPLETE")
        print(f"  Total distance: {total_distance:.0f}m ({total_distance/1000:.2f}km)")
        print(f"  Segments walked: {self.segments_walked}")
        print("  All segments recorded to database")
        print("=" * 60)

    def _count_segments_through(self, target_route_index: int):
        """Count distance and record segments from _last_counted_index to target_route_index."""
        route = self.planner.planned_route
        for idx in range(self._last_counted_index, target_route_index):
            seg = self.graph.get_segment(route[idx], route[idx + 1])
            if seg:
                self.history.record_segment(seg.id)
                self.planner.walked_distance += seg.length
                self.segments_walked += 1
                self.logger.log("Walked segment", {
                    "segment": seg.id,
                    "length": seg.length,
                    "total": self.planner.walked_distance,
                    "name": seg.name
                })
        self._last_counted_index = target_route_index

    def _get_next_waypoint(self) -> Optional[int]:
        """Get next waypoint, skipping any that are too close to current location."""
        if not self.current_location or not self.planner or not self.graph:
            return self.planner.get_next_planned_node() if self.planner else None

        min_dist = CONFIG["min_waypoint_distance"]

        while True:
            next_node = self.planner.get_next_planned_node()
            if not next_node:
                return None

            next_loc = self.graph.get_node_location(next_node)
            if not next_loc:
                return next_node

            dist = haversine_distance(
                self.current_location.lat, self.current_location.lon,
                next_loc[0], next_loc[1]
            )

            if dist >= min_dist:
                return next_node

            # Too close, skip this waypoint
            self.planner.advance_route()

    def _get_direction_text(self) -> Optional[str]:
        """Get direction text for next waypoint: '{compass}, {distance}'"""
        if not self.next_node or not self.current_location or not self.graph:
            return None
        next_loc = self.graph.get_node_location(self.next_node)
        if not next_loc:
            return None
        dist_to_next = haversine_distance(
            self.current_location.lat, self.current_location.lon,
            next_loc[0], next_loc[1]
        )
        bearing = bearing_between(
            self.current_location.lat, self.current_location.lon,
            next_loc[0], next_loc[1]
        )
        compass = bearing_to_compass(bearing)
        return f"{compass}, {int(dist_to_next)}"

    def _check_distance_milestone(self) -> Optional[int]:
        """Check if a distance milestone was crossed. Returns total distance if so."""
        if not self.planner:
            return None
        walked = int(self.planner.walked_distance)
        interval = CONFIG["distance_milestone_interval"]
        current_milestone = (walked // interval) * interval
        if current_milestone > self.last_distance_milestone and current_milestone > 0:
            self.last_distance_milestone = current_milestone
            return walked
        return None

    def _speak_waypoint_reached(self):
        """Speak waypoint arrival: tone + direction + optional distance milestone"""
        direction = self._get_direction_text()
        milestone = self._check_distance_milestone()

        parts = []
        if direction:
            parts.append(direction)
        if milestone is not None:
            parts.append(str(milestone))

        if parts:
            self.audio.tone()
            message = ". ".join(parts)
            self.audio.speak(message)
            self.logger.log(f"AUDIO: {message}")

    def _speak_status(self):
        """Speak direction to next waypoint (no tone). Used for initial direction and debug-gui."""
        direction = self._get_direction_text()
        if direction:
            self.audio.speak(direction)
            self.logger.log(f"AUDIO: {direction}")

    def update(self) -> bool:
        """Main update loop - returns False when walk is complete"""

        # Periodic status updates
        self.periodic_update()

        # Get current GPS
        location = self.gps_source.get_location()
        if not location:
            self.logger.log("GPS fix failed", {"status": self.gps_source.get_status() if hasattr(self.gps_source, 'get_status') else "unknown"})
            return True  # Keep going even with GPS errors

        self.current_location = location

        # Send state update to debug server
        if self.debug_server:
            self.debug_server.send_state(self.get_state())

        # Find nearest node to current GPS position
        nearest = self.graph.find_nearest_node(location.lat, location.lon)
        if not nearest:
            return True

        nearest_loc = self.graph.get_node_location(nearest)
        dist_to_nearest = haversine_distance(
            location.lat, location.lon, nearest_loc[0], nearest_loc[1]
        )

        # Calculate distance to next planned node
        dist_to_next = float("inf")
        if self.next_node:
            next_loc = self.graph.get_node_location(self.next_node)
            dist_to_next = haversine_distance(
                location.lat, location.lon, next_loc[0], next_loc[1]
            )

        # Check for route deviation (temporarily disabled)
        # deviation_detected = self._check_route_deviation(location, nearest, dist_to_nearest)
        deviation_detected = False

        # Determine what happened:
        # 1. Arrived at expected next_node (on planned route)
        # 2. Deviated too far from route (needs recalculation) - disabled
        # 3. Still at current node (no movement)

        arrived_at_next = self.next_node and dist_to_next < CONFIG["intersection_arrival_radius"]

        if arrived_at_next:
            # Count all segments from last counted position to arrival
            # next_node is always at route_index + 1 after _get_next_waypoint
            arrival_index = self.planner.route_index + 1
            self._count_segments_through(arrival_index)
            print(f"Total: {self.planner.walked_distance:.0f}m")

            # Check if walk is complete (arrived back at start)
            if (self.next_node == self.planner.start_node and
                self.planner.walked_distance >= self.planner.target_distance * 0.8):
                self.audio.speak("Walk complete")
                self.logger.log("Walk complete")
                print("Walk complete!")
                return False

            # Advance along planned route
            self.previous_node = self.current_node
            self.current_node = self.next_node
            self.planner.advance_route()
            self.next_node = self._get_next_waypoint()

            # Send updated state to debug GUI immediately
            if self.debug_server:
                self.debug_server.send_state(self.get_state())

            # Immediate audio update with directions to next waypoint
            if self.next_node:
                self._speak_waypoint_reached()

            if not self.next_node:
                # Count any remaining segments to end of route
                self._count_segments_through(len(self.planner.planned_route) - 1)
                # Reached end of planned route
                if self.current_node == self.planner.start_node:
                    self.audio.speak("Walk complete")
                    self.logger.log("Walk complete")
                    print("Walk complete!")
                else:
                    self.logger.log("End of planned route")
                    print("End of planned route")
                return False

        # In debug-gui mode, give immediate audio feedback after each location update
        if self.debug_gui:
            self._speak_status()

        return True

    def _check_route_deviation(self, location: Location, nearest_node: int,
                                dist_to_nearest: float) -> bool:
        """Check if user has deviated too far from the planned route.

        Returns True if recalculation is needed.
        """
        if not self.planner.planned_route:
            return False

        # Check if nearest node is on the planned route ahead
        if self.planner.is_on_route(nearest_node):
            return False

        # Check distance from GPS position to the expected next node
        if self.next_node:
            next_loc = self.graph.get_node_location(self.next_node)
            if next_loc:
                dist_to_next = haversine_distance(
                    location.lat, location.lon, next_loc[0], next_loc[1]
                )
                # If we're far from both the expected path and any known node
                if dist_to_next > CONFIG["route_deviation_threshold"]:
                    # Confirm we've actually moved to a different node
                    if nearest_node != self.current_node and dist_to_nearest < CONFIG["intersection_arrival_radius"]:
                        return True

        return False

    def _recalculate_route(self):
        """Recalculate route from current position"""
        remaining_distance = self.planner.target_distance - self.planner.walked_distance
        print(f"Recalculating route for remaining {remaining_distance:.0f}m...")

        # Store original start for return
        original_start = self.planner.start_node

        # Calculate new route
        new_route = self.planner.calculate_full_route(self.current_node, remaining_distance)

        # Restore original start for proper loop completion
        self.planner.start_node = original_start

        self.logger.log("Route recalculated", {
            "new_nodes": len(new_route),
            "remaining_distance": remaining_distance
        })
        print(f"New route: {len(new_route)} nodes")

        # Get next waypoint
        self.next_node = self.planner.get_next_planned_node()

        if self.next_node and self.current_location:
            next_loc = self.graph.get_node_location(self.next_node)
            if next_loc:
                dist_to_next = haversine_distance(
                    self.current_location.lat, self.current_location.lon,
                    next_loc[0], next_loc[1]
                )
                bearing = bearing_between(
                    self.current_location.lat, self.current_location.lon,
                    next_loc[0], next_loc[1]
                )
                compass = bearing_to_compass(bearing)
                instruction = f"Recalculating. {int(dist_to_next)} {compass}"
                self.audio.speak(instruction)
                self.logger.log(f"Recalculated: {instruction}")
                print(f"Recalculated: {instruction}")

    def get_poll_interval(self) -> float:
        """Get poll interval, respecting playback speed if applicable"""
        if isinstance(self.gps_source, GPSPlayback):
            return self.gps_source.get_poll_interval()
        return CONFIG["gps_poll_interval"]

    def is_playback_finished(self) -> bool:
        """Check if playback is complete"""
        if isinstance(self.gps_source, GPSPlayback):
            return self.gps_source.is_finished()
        return False

    def run(self, target_distance: float):
        """Run the walk"""

        print(f"\n=== Walker ===")
        print(f"Target distance: {target_distance}m")
        if self.preview_mode:
            print("Mode: PREVIEW (calculate and display route)")
        elif self.execute_mode:
            print("Mode: EXECUTE (virtual walk for testing)")
        else:
            if isinstance(self.gps_source, GPSPlayback):
                print(f"Playback mode: {self.gps_source.speed}x speed")
            print("Press Ctrl+C to stop")
        print()

        if not self.initialize(target_distance):
            # Preview mode exits after displaying route
            if self.preview_mode:
                self.history.close()
                self.logger.close()
            return

        # Execute mode: run through route virtually
        if self.execute_mode:
            self.execute_route()
            self.history.close()
            self.logger.close()
            return

        # Normal walking mode
        try:
            while self.update():
                # Check if playback finished
                if self.is_playback_finished():
                    print("\nPlayback finished")
                    self.logger.log("Playback finished")
                    break
                time.sleep(self.get_poll_interval())
        except KeyboardInterrupt:
            print("\nWalk interrupted")
            self.audio.speak("Walk ended")
            self.logger.log("Walk interrupted by user")
        finally:
            # Record walk stats
            if self.walk_id:
                self.history.end_walk(
                    self.walk_id,
                    self.planner.walked_distance if self.planner else 0,
                    self.segments_walked
                )

            # Save GPS recording if applicable
            if isinstance(self.gps_source, GPSRecorder):
                self.gps_source.save()

            summary = {
                "distance": self.planner.walked_distance if self.planner else 0,
                "segments": self.segments_walked,
                "duration": time.time() - self.walk_start_time if self.walk_start_time else 0
            }
            self.logger.log("Walk summary", summary)

            print(f"\nWalk summary:")
            print(f"  Distance: {summary['distance']:.0f}m")
            print(f"  Segments: {summary['segments']}")
            print(f"  Duration: {summary['duration']/60:.1f} minutes")

            self.history.close()
            self.logger.close()
