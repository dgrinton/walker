#!/usr/bin/env python3
"""
Walker - Turn-by-turn pedestrian exploration guide

Usage:
    python -m walker [distance_km] [options]

Options:
    --record FILE     Record GPS trace to JSON file for debugging
    --playback FILE   Playback GPS trace from JSON file
    --speed FACTOR    Playback speed multiplier (default: 1.0)
    --preview         Preview the calculated route without walking
    --execute         Execute route virtually and add to database (testing)
    --lat LAT         Starting latitude (for testing without GPS)
    --lon LON         Starting longitude (for testing without GPS)
    --html FILE       Output route visualization to HTML file (preview mode)
    --gpx FILE        Export route to GPX file (preview mode)
    --debug-gui       Run with web-based visual debugger (requires --lat/--lon)
    --edit-zones      Open the exclusion zone editor
    --reset           Erase all walked segment history and exit
    --debug-parallels Interactive map of parallel/corridor exclusions (requires --lat/--lon)
"""

import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from .app import Walker
from .gps import GPSRecorder, GPSPlayback
from .debug_gui import WebSocketGPS


def _debug_parallels(lat: float, lon: float):
    """Generate an interactive HTML map for debugging parallel/corridor exclusions."""
    from .osm import OSMFetcher
    from .graph import StreetGraph

    # Fetch OSM data and build graph
    osm_data = OSMFetcher.fetch_streets(lat, lon, radius=500)
    graph = StreetGraph()
    graph.build_from_osm(osm_data)

    # Collect segment data with coordinates
    segments_data = []
    for seg in graph.segments.values():
        loc1 = graph.nodes.get(seg.node1)
        loc2 = graph.nodes.get(seg.node2)
        if not loc1 or not loc2:
            continue
        segments_data.append({
            "id": seg.id,
            "coords": [[loc1[0], loc1[1]], [loc2[0], loc2[1]]],
            "name": seg.name or "",
            "roadType": seg.road_type,
            "length": round(seg.length, 1),
        })

    # Build exclusion map: for each segment, union of parallel + corridor siblings
    exclusions = {}
    for seg_id in graph.segments:
        parallel = sorted(graph.parallel_segments.get(seg_id, set()))
        corridor = set()
        if seg_id in graph.corridor_groups:
            group_id = graph.corridor_groups[seg_id]
            corridor = graph.corridor_members[group_id] - {seg_id}
        corridor = sorted(corridor)
        if parallel or corridor:
            exclusions[seg_id] = {"parallel": parallel, "corridor": corridor}

    # Stats
    total_segments = len(segments_data)
    segments_with_parallels = sum(1 for s in graph.segments if s in graph.parallel_segments)
    corridor_group_count = len(graph.corridor_members)

    segments_json = json.dumps(segments_data)
    exclusions_json = json.dumps(exclusions)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Parallel Segment Debugger</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        #map {{ position: absolute; top: 0; bottom: 0; left: 0; right: 320px; }}
        #sidebar {{ position: absolute; top: 0; bottom: 0; right: 0; width: 320px; background: #1a1a2e; color: #eee; overflow-y: auto; padding: 15px; box-sizing: border-box; }}
        h2 {{ margin-top: 0; color: #fff; }}
        .stat {{ margin: 8px 0; padding: 10px; background: #16213e; border-radius: 5px; }}
        .stat-label {{ font-size: 12px; color: #999; }}
        .stat-value {{ font-size: 20px; font-weight: bold; color: #fff; }}
        .legend {{ margin-top: 16px; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; font-size: 13px; }}
        .legend-color {{ width: 30px; height: 4px; margin-right: 10px; border-radius: 2px; }}
        #selection-info {{ margin-top: 16px; padding: 12px; background: #16213e; border-radius: 5px; display: none; }}
        #selection-info h3 {{ margin: 0 0 8px 0; color: #fff; font-size: 14px; }}
        .info-row {{ font-size: 13px; margin: 4px 0; }}
        .info-label {{ color: #999; }}
        .info-value {{ color: #fff; }}
        .exclusion-counts {{ margin-top: 10px; }}
        .exclusion-count {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; margin-right: 6px; font-weight: bold; }}
        .exc-parallel {{ background: #a855f7; color: white; }}
        .exc-corridor {{ background: #eab308; color: black; }}
        .hint {{ margin-top: 16px; font-size: 12px; color: #666; text-align: center; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="sidebar">
        <h2>Parallel Segment Debugger</h2>
        <div class="stat">
            <div class="stat-label">Total Segments</div>
            <div class="stat-value">{total_segments}</div>
        </div>
        <div class="stat">
            <div class="stat-label">With Geometric Parallels</div>
            <div class="stat-value">{segments_with_parallels}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Name Corridor Groups</div>
            <div class="stat-value">{corridor_group_count}</div>
        </div>
        <div class="legend">
            <h3 style="font-size: 13px; color: #999; margin: 0 0 6px 0;">Road Types</h3>
            <div class="legend-item"><div class="legend-color" style="background: #22c55e;"></div> footway / path / pedestrian</div>
            <div class="legend-item"><div class="legend-color" style="background: #3b82f6;"></div> residential / living_street</div>
            <div class="legend-item"><div class="legend-color" style="background: #f97316;"></div> secondary / tertiary</div>
            <div class="legend-item"><div class="legend-color" style="background: #9ca3af;"></div> service / unclassified</div>
            <div class="legend-item"><div class="legend-color" style="background: #ef4444;"></div> primary / trunk</div>
        </div>
        <div class="legend" style="margin-top: 12px;">
            <h3 style="font-size: 13px; color: #999; margin: 0 0 6px 0;">Exclusions</h3>
            <div class="legend-item"><div class="legend-color" style="background: #a855f7;"></div> Geometric parallel</div>
            <div class="legend-item"><div class="legend-color" style="background: #eab308;"></div> Name corridor</div>
        </div>
        <div id="selection-info">
            <h3 id="sel-name">—</h3>
            <div class="info-row"><span class="info-label">Road type: </span><span class="info-value" id="sel-type"></span></div>
            <div class="info-row"><span class="info-label">Length: </span><span class="info-value" id="sel-length"></span></div>
            <div class="info-row"><span class="info-label">Segment ID: </span><span class="info-value" id="sel-id" style="font-size: 11px; word-break: break-all;"></span></div>
            <div class="exclusion-counts" id="sel-counts"></div>
        </div>
        <div class="hint">Click a segment to see exclusions. Click background to reset.</div>
    </div>
    <script>
        var segments = {segments_json};
        var exclusions = {exclusions_json};

        var map = L.map('map').setView([{lat}, {lon}], 16);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        // Color by road type
        function roadColor(rt) {{
            if (['footway','path','pedestrian'].includes(rt)) return '#22c55e';
            if (['residential','living_street'].includes(rt)) return '#3b82f6';
            if (['secondary','tertiary'].includes(rt)) return '#f97316';
            if (['primary','trunk'].includes(rt)) return '#ef4444';
            return '#9ca3af';
        }}

        // Build segment lines
        var segMap = {{}};  // id -> {{line, data}}
        var allLines = [];
        segments.forEach(function(seg) {{
            var line = L.polyline(seg.coords, {{
                color: roadColor(seg.roadType),
                weight: 4,
                opacity: 0.7
            }}).addTo(map);
            line._segId = seg.id;
            segMap[seg.id] = {{line: line, data: seg}};
            allLines.push(line);

            line.on('click', function(e) {{
                L.DomEvent.stopPropagation(e);
                selectSegment(seg.id);
            }});
        }});

        // Fit bounds
        if (allLines.length > 0) {{
            var group = L.featureGroup(allLines);
            map.fitBounds(group.getBounds(), {{padding: [20, 20]}});
        }}

        var selected = null;

        function resetView() {{
            selected = null;
            allLines.forEach(function(line) {{
                var seg = segMap[line._segId];
                line.setStyle({{
                    color: roadColor(seg.data.roadType),
                    weight: 4,
                    opacity: 0.7
                }});
            }});
            document.getElementById('selection-info').style.display = 'none';
        }}

        function selectSegment(segId) {{
            selected = segId;
            var exc = exclusions[segId] || {{parallel: [], corridor: []}};
            var parallelSet = new Set(exc.parallel || []);
            var corridorSet = new Set(exc.corridor || []);
            var excludedSet = new Set([...parallelSet, ...corridorSet]);

            // Dim everything, then highlight
            allLines.forEach(function(line) {{
                var id = line._segId;
                if (id === segId) {{
                    line.setStyle({{ color: '#ef4444', weight: 6, opacity: 1 }});
                    line.bringToFront();
                }} else if (parallelSet.has(id) && corridorSet.has(id)) {{
                    // Both: show as parallel (magenta) with higher priority
                    line.setStyle({{ color: '#a855f7', weight: 5, opacity: 0.9 }});
                    line.bringToFront();
                }} else if (parallelSet.has(id)) {{
                    line.setStyle({{ color: '#a855f7', weight: 5, opacity: 0.9 }});
                    line.bringToFront();
                }} else if (corridorSet.has(id)) {{
                    line.setStyle({{ color: '#eab308', weight: 5, opacity: 0.9 }});
                    line.bringToFront();
                }} else {{
                    line.setStyle({{ color: roadColor(segMap[id].data.roadType), weight: 3, opacity: 0.15 }});
                }}
            }});

            // Update sidebar
            var seg = segMap[segId].data;
            document.getElementById('sel-name').textContent = seg.name || '(unnamed)';
            document.getElementById('sel-type').textContent = seg.roadType;
            document.getElementById('sel-length').textContent = seg.length + ' m';
            document.getElementById('sel-id').textContent = seg.id;

            var countsHtml = '';
            if (parallelSet.size > 0) {{
                countsHtml += '<span class="exclusion-count exc-parallel">' + parallelSet.size + ' geometric</span>';
            }}
            if (corridorSet.size > 0) {{
                countsHtml += '<span class="exclusion-count exc-corridor">' + corridorSet.size + ' corridor</span>';
            }}
            if (!parallelSet.size && !corridorSet.size) {{
                countsHtml = '<span style="color: #666; font-size: 12px;">No exclusions</span>';
            }}
            document.getElementById('sel-counts').innerHTML = countsHtml;
            document.getElementById('selection-info').style.display = 'block';
        }}

        map.on('click', function() {{
            resetView();
        }});
    </script>
</body>
</html>'''

    output_file = "debug_parallels.html"
    with open(output_file, 'w') as f:
        f.write(html)

    abs_path = os.path.abspath(output_file)
    print(f"\nParallel debug map saved to: {abs_path}")
    webbrowser.open(f"file://{abs_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Walker - Turn-by-turn pedestrian exploration guide"
    )
    parser.add_argument("distance", type=float, nargs="?", default=2.0,
                        help="Target distance in km (default: 2.0)")
    parser.add_argument("--record", metavar="FILE",
                        help="Record GPS trace to JSON file")
    parser.add_argument("--playback", metavar="FILE",
                        help="Playback GPS trace from JSON file")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--log", metavar="FILE",
                        help="Log file path (default: walker_TIMESTAMP.log)")
    parser.add_argument("--preview", action="store_true",
                        help="Preview the calculated route without walking")
    parser.add_argument("--execute", action="store_true",
                        help="Execute route virtually and add to database (testing)")
    parser.add_argument("--lat", type=float, metavar="LAT",
                        help="Starting latitude (for testing without GPS)")
    parser.add_argument("--lon", type=float, metavar="LON",
                        help="Starting longitude (for testing without GPS)")
    parser.add_argument("--html", metavar="FILE",
                        help="Output route visualization to HTML file (preview mode)")
    parser.add_argument("--gpx", metavar="FILE",
                        help="Export route to GPX file (preview mode)")
    parser.add_argument("--debug-gui", action="store_true",
                        help="Run with web-based visual debugger (requires --lat and --lon)")
    parser.add_argument("--edit-zones", action="store_true",
                        help="Open the exclusion zone editor")
    parser.add_argument("--reset", action="store_true",
                        help="Erase all walked segment history and exit")
    parser.add_argument("--debug-parallels", action="store_true",
                        help="Generate interactive map showing parallel/corridor exclusions (requires --lat/--lon)")

    args = parser.parse_args()

    # Validate lat/lon - must provide both or neither
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be used together")

    # Validate debug-gui requires lat/lon
    if args.debug_gui and (args.lat is None or args.lon is None):
        parser.error("--debug-gui requires --lat and --lon")

    # Validate debug-parallels requires lat/lon
    if args.debug_parallels and (args.lat is None or args.lon is None):
        parser.error("--debug-parallels requires --lat and --lon")

    # Reset history: early exit
    if args.reset:
        from .history import HistoryDB
        history = HistoryDB()
        cursor = history.conn.execute("SELECT COUNT(*) FROM segment_history")
        segment_count = cursor.fetchone()[0]
        cursor = history.conn.execute("SELECT COUNT(*) FROM walks")
        walk_count = cursor.fetchone()[0]
        history.reset_history()
        print(f"Cleared {segment_count} segment records and {walk_count} walk sessions.")
        print("Exclusion zones preserved.")
        history.close()
        return

    # Zone editor: early exit before creating Walker
    if args.edit_zones:
        from .zone_editor import ZoneEditorServer
        from .history import HistoryDB
        history = HistoryDB()
        center = (args.lat, args.lon) if args.lat is not None else None
        server = ZoneEditorServer(history, center=center)
        server.start()
        return

    # Debug parallels: early exit
    if args.debug_parallels:
        _debug_parallels(args.lat, args.lon)
        return

    # Convert km to meters
    distance = args.distance * 1000

    # Determine log path
    log_path = args.log
    if not log_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"walker_{timestamp}.log"

    # Create walker
    start_location = (args.lat, args.lon) if args.lat is not None else None
    walker = Walker(
        log_path=log_path,
        preview_mode=args.preview,
        execute_mode=args.execute,
        start_location=start_location,
        html_output=args.html,
        gpx_output=args.gpx,
        debug_gui=args.debug_gui
    )

    # Set up GPS source
    if args.debug_gui:
        # Debug GUI uses WebSocketGPS for click-based location
        walker.set_gps_source(WebSocketGPS(walker.debug_server))
    elif args.playback:
        if not Path(args.playback).exists():
            print(f"Playback file not found: {args.playback}")
            sys.exit(1)
        walker.set_gps_source(GPSPlayback(args.playback, args.speed))
    elif args.record:
        walker.set_gps_source(GPSRecorder(walker.gps, args.record))

    walker.run(distance)


if __name__ == "__main__":
    main()
