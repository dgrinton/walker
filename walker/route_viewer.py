"""Interactive HTML viewer for route log JSON files.

Usage:
    python -m walker.route_viewer routes/route_001.json
    python -m walker.route_viewer routes/route_001.json --output route.html
"""

import json
import os
import sys
import webbrowser


def generate_html(route_data: dict) -> str:
    """Generate interactive HTML from route log data."""
    steps = route_data.get("steps", [])
    return_path = route_data.get("return_path")
    start = route_data.get("start_location", {})
    target_dist = route_data.get("target_distance_m", 0)
    actual_dist = route_data.get("actual_distance_m", 0)
    graph_stats = route_data.get("graph_stats", {})

    if not steps:
        return "<html><body><p>No steps in route log.</p></body></html>"

    # Compute center from all step locations
    lats = []
    lons = []
    for s in steps:
        lats.append(s["from_location"]["lat"])
        lons.append(s["from_location"]["lon"])
    lats.append(steps[-1]["to_location"]["lat"])
    lons.append(steps[-1]["to_location"]["lon"])
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Determine return path node set for coloring
    return_step_start = return_path["triggered_at_step"] if return_path else len(steps)

    # Count stats
    total_steps = len(steps)
    new_segments = sum(1 for s in steps if s["decision"]["score_breakdown"].get("novelty_factor", 0) == 0)
    walked_segments = total_steps - new_segments

    steps_json = json.dumps(steps)
    return_json = json.dumps(return_path) if return_path else "null"

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Route Viewer</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        #map {{ position: absolute; top: 0; bottom: 0; left: 0; right: 340px; }}
        #sidebar {{ position: absolute; top: 0; bottom: 0; right: 0; width: 340px; background: #1a1a2e; color: #eee; overflow-y: auto; padding: 15px; box-sizing: border-box; }}
        h2 {{ margin-top: 0; color: #fff; }}
        .stat {{ margin: 8px 0; padding: 10px; background: #16213e; border-radius: 5px; }}
        .stat-label {{ font-size: 12px; color: #999; }}
        .stat-value {{ font-size: 20px; font-weight: bold; color: #fff; }}
        .stat-row {{ display: flex; gap: 8px; }}
        .stat-row .stat {{ flex: 1; }}
        .legend {{ margin-top: 12px; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; font-size: 13px; }}
        .legend-color {{ width: 30px; height: 4px; margin-right: 10px; border-radius: 2px; }}
        #step-detail {{ margin-top: 12px; padding: 12px; background: #16213e; border-radius: 5px; display: none; }}
        #step-detail h3 {{ margin: 0 0 8px 0; color: #fff; font-size: 14px; }}
        .detail-row {{ font-size: 13px; margin: 3px 0; }}
        .detail-label {{ color: #999; }}
        .detail-value {{ color: #fff; }}
        .breakdown {{ margin-top: 8px; }}
        .breakdown-item {{ display: flex; justify-content: space-between; font-size: 12px; padding: 2px 0; }}
        .breakdown-item .label {{ color: #999; }}
        .breakdown-item .value {{ color: #fff; font-weight: bold; }}
        .alternatives {{ margin-top: 10px; }}
        .alt-item {{ padding: 6px 8px; margin: 4px 0; background: #0f3460; border-radius: 4px; font-size: 12px; }}
        .alt-rejected {{ background: #3d1f1f; }}
        .alt-score {{ float: right; font-weight: bold; }}
        .hint {{ margin-top: 12px; font-size: 12px; color: #555; text-align: center; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="sidebar">
        <h2>Route Viewer</h2>
        <div class="stat-row">
            <div class="stat">
                <div class="stat-label">Target</div>
                <div class="stat-value">{target_dist/1000:.2f} km</div>
            </div>
            <div class="stat">
                <div class="stat-label">Actual</div>
                <div class="stat-value">{actual_dist/1000:.2f} km</div>
            </div>
        </div>
        <div class="stat-row">
            <div class="stat">
                <div class="stat-label">Steps</div>
                <div class="stat-value">{total_steps}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Novelty</div>
                <div class="stat-value">{new_segments / total_steps * 100 if total_steps else 0:.0f}%</div>
            </div>
        </div>
        <div class="stat-row">
            <div class="stat">
                <div class="stat-label">New</div>
                <div class="stat-value" style="color: #22c55e;">{new_segments}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Walked</div>
                <div class="stat-value" style="color: #3b82f6;">{walked_segments}</div>
            </div>
        </div>
        <div class="stat">
            <div class="stat-label">Graph</div>
            <div class="stat-value" style="font-size: 14px;">{graph_stats.get('total_nodes', '?')} nodes, {graph_stats.get('total_segments', '?')} segs</div>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background: #22c55e;"></div> New segment</div>
            <div class="legend-item"><div class="legend-color" style="background: #3b82f6;"></div> Previously walked</div>
            <div class="legend-item"><div class="legend-color" style="background: #1a6b3a;"></div> Return path (new)</div>
            <div class="legend-item"><div class="legend-color" style="background: #1e4a8a;"></div> Return path (walked)</div>
        </div>
        <div id="step-detail">
            <h3 id="sd-title">Step</h3>
            <div class="detail-row"><span class="detail-label">Segment: </span><span class="detail-value" id="sd-name"></span></div>
            <div class="detail-row"><span class="detail-label">Type: </span><span class="detail-value" id="sd-type"></span></div>
            <div class="detail-row"><span class="detail-label">Length: </span><span class="detail-value" id="sd-length"></span></div>
            <div class="detail-row"><span class="detail-label">Cumulative: </span><span class="detail-value" id="sd-cumul"></span></div>
            <div class="detail-row"><span class="detail-label">Bearing: </span><span class="detail-value" id="sd-bearing"></span></div>
            <div class="detail-row"><span class="detail-label">Turn: </span><span class="detail-value" id="sd-turn"></span></div>
            <div class="detail-row"><span class="detail-label">Dist to start: </span><span class="detail-value" id="sd-dist"></span></div>
            <div class="detail-row"><span class="detail-label">Budget left: </span><span class="detail-value" id="sd-budget"></span></div>
            <div class="detail-row"><span class="detail-label">Busy adj: </span><span class="detail-value" id="sd-busy"></span></div>
            <div class="breakdown" id="sd-breakdown"></div>
            <div class="alternatives" id="sd-alts"></div>
        </div>
        <div class="hint">Click a segment for details</div>
    </div>
    <script>
        var steps = {steps_json};
        var returnPath = {return_json};
        var returnStart = {return_step_start};

        var map = L.map('map').setView([{center_lat}, {center_lon}], 15);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap'
        }}).addTo(map);

        var bounds = [];
        var lines = [];

        steps.forEach(function(step, idx) {{
            var isReturn = idx >= returnStart;
            var isNew = step.decision.score_breakdown.novelty_factor === 0;
            var color;
            if (isReturn && isNew) color = '#1a6b3a';
            else if (isReturn) color = '#1e4a8a';
            else if (isNew) color = '#22c55e';
            else color = '#3b82f6';

            var coords = [
                [step.from_location.lat, step.from_location.lon],
                [step.to_location.lat, step.to_location.lon]
            ];
            var line = L.polyline(coords, {{
                color: color,
                weight: 5,
                opacity: 0.85
            }}).addTo(map);
            line._stepIdx = idx;
            bounds.push(coords[0]);
            bounds.push(coords[1]);

            line.on('click', function(e) {{
                L.DomEvent.stopPropagation(e);
                showStep(idx);
            }});
            lines.push(line);
        }});

        // Start marker
        if (steps.length > 0) {{
            var s0 = steps[0];
            L.circleMarker([s0.from_location.lat, s0.from_location.lon], {{
                radius: 10, fillColor: '#ef4444', color: '#fff', weight: 2, fillOpacity: 1
            }}).addTo(map).bindPopup('Start');
        }}

        // Step number markers at intersections (every 5 steps)
        steps.forEach(function(step, idx) {{
            if (idx % 5 === 0 && idx > 0) {{
                L.circleMarker([step.from_location.lat, step.from_location.lon], {{
                    radius: 4, fillColor: '#fff', color: '#333', weight: 1, fillOpacity: 0.8
                }}).addTo(map).bindTooltip('' + idx, {{permanent: true, direction: 'top', className: 'step-label'}});
            }}
        }});

        if (bounds.length > 0) map.fitBounds(bounds, {{padding: [30, 30]}});

        function showStep(idx) {{
            var step = steps[idx];
            var d = step.decision;
            var bd = d.score_breakdown;

            document.getElementById('sd-title').textContent = 'Step ' + step.step_index + ' (score: ' + d.chosen_score + ')';
            document.getElementById('sd-name').textContent = step.segment_name || '(unnamed)';
            document.getElementById('sd-type').textContent = step.road_type;
            document.getElementById('sd-length').textContent = step.segment_length_m + ' m';
            document.getElementById('sd-cumul').textContent = step.cumulative_distance_m + ' m';
            document.getElementById('sd-bearing').textContent = step.bearing + '\u00b0';
            document.getElementById('sd-turn').textContent = step.turn_angle !== null ? step.turn_angle + '\u00b0' : 'n/a';
            document.getElementById('sd-dist').textContent = step.context.dist_to_start_m + ' m';
            document.getElementById('sd-budget').textContent = step.context.remaining_budget_m + ' m';
            document.getElementById('sd-busy').textContent = step.busy_road_adjacent ? 'Yes' : 'No';

            // Breakdown
            var bdHtml = '<h4 style="margin:4px 0;font-size:12px;color:#999;">Score Breakdown</h4>';
            for (var key in bd) {{
                bdHtml += '<div class="breakdown-item"><span class="label">' + key + '</span><span class="value">' + bd[key] + '</span></div>';
            }}
            document.getElementById('sd-breakdown').innerHTML = bdHtml;

            // Alternatives
            var alts = d.alternatives || [];
            var altHtml = '<h4 style="margin:4px 0;font-size:12px;color:#999;">Alternatives (' + alts.length + ')</h4>';
            alts.forEach(function(alt) {{
                var cls = alt.rejection_reason ? 'alt-item alt-rejected' : 'alt-item';
                var name = alt.segment_name || '(unnamed)';
                var scoreText = alt.rejection_reason ? alt.rejection_reason : ('score: ' + alt.score);
                altHtml += '<div class="' + cls + '">' + name + ' <span class="alt-score">' + scoreText + '</span>';
                if (alt.score_breakdown) {{
                    altHtml += '<div style="margin-top:3px;font-size:11px;color:#888;">';
                    for (var k in alt.score_breakdown) {{
                        altHtml += k + ':' + alt.score_breakdown[k] + ' ';
                    }}
                    altHtml += '</div>';
                }}
                altHtml += '</div>';
            }});
            document.getElementById('sd-alts').innerHTML = altHtml;

            document.getElementById('step-detail').style.display = 'block';

            // Highlight selected line
            lines.forEach(function(line, i) {{
                if (i === idx) {{
                    line.setStyle({{ weight: 8, opacity: 1 }});
                    line.bringToFront();
                }} else {{
                    line.setStyle({{ weight: 5, opacity: 0.85 }});
                }}
            }});
        }}

        map.on('click', function() {{
            document.getElementById('step-detail').style.display = 'none';
            lines.forEach(function(line) {{
                line.setStyle({{ weight: 5, opacity: 0.85 }});
            }});
        }});
    </script>
</body>
</html>'''
    return html


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m walker.route_viewer ROUTE_JSON [--output FILE]")
        sys.exit(1)

    route_file = sys.argv[1]
    output_file = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]

    with open(route_file) as f:
        route_data = json.load(f)

    html = generate_html(route_data)

    if output_file is None:
        output_file = os.path.splitext(route_file)[0] + ".html"

    with open(output_file, "w") as f:
        f.write(html)

    abs_path = os.path.abspath(output_file)
    print(f"Route viewer saved to: {abs_path}")
    webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
