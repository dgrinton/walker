#!/usr/bin/env python3
"""
Visualize a recorded GPS trace on a map.

Usage:
    python visualize_trace.py trace.json [--output map.html]
"""

import argparse
import json
from pathlib import Path

import folium
from folium import plugins


def load_trace(trace_path: str) -> list[dict]:
    """Load GPS trace from JSON file"""
    with open(trace_path) as f:
        data = json.load(f)
    return data["trace"]


def create_trace_map(trace: list[dict], output_path: str):
    """Create map visualization of GPS trace"""

    # Filter to only entries with valid locations
    valid_entries = [e for e in trace if e.get("location")]

    if not valid_entries:
        print("No valid GPS locations in trace")
        return

    # Get center point
    lats = [e["location"]["lat"] for e in valid_entries]
    lons = [e["location"]["lon"] for e in valid_entries]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16)

    # Add tile options
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

    # Create path coordinates
    path_coords = [[e["location"]["lat"], e["location"]["lon"]] for e in valid_entries]

    # Add path as polyline
    folium.PolyLine(
        path_coords,
        weight=4,
        color="blue",
        opacity=0.7,
        popup="GPS Trace"
    ).add_to(m)

    # Add markers for each point with info
    points_group = folium.FeatureGroup(name="GPS Points", show=False)

    for i, entry in enumerate(valid_entries):
        loc = entry["location"]
        elapsed = entry.get("elapsed", 0)
        accuracy = loc.get("accuracy", "unknown")
        status = entry.get("status", "")

        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        popup = f"""
            <b>Point {i + 1}</b><br>
            Time: {minutes}m {seconds}s<br>
            Lat: {loc['lat']:.6f}<br>
            Lon: {loc['lon']:.6f}<br>
            Accuracy: {accuracy}m<br>
            Status: {status}
        """

        # Color based on accuracy
        if accuracy and accuracy != "unknown":
            if accuracy < 10:
                color = "green"
            elif accuracy < 20:
                color = "orange"
            else:
                color = "red"
        else:
            color = "gray"

        folium.CircleMarker(
            location=[loc["lat"], loc["lon"]],
            radius=5,
            color=color,
            fill=True,
            popup=folium.Popup(popup, max_width=200)
        ).add_to(points_group)

    points_group.add_to(m)

    # Add start marker
    if valid_entries:
        start = valid_entries[0]["location"]
        folium.Marker(
            [start["lat"], start["lon"]],
            popup="Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)

        # Add end marker
        end = valid_entries[-1]["location"]
        folium.Marker(
            [end["lat"], end["lon"]],
            popup="End",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)

    # Mark GPS failures
    failures_group = folium.FeatureGroup(name="GPS Failures", show=True)
    last_valid = None

    for i, entry in enumerate(trace):
        if entry.get("location"):
            last_valid = entry["location"]
        elif last_valid:
            # GPS failure - mark at last known position
            elapsed = entry.get("elapsed", 0)
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            folium.CircleMarker(
                location=[last_valid["lat"], last_valid["lon"]],
                radius=8,
                color="red",
                fill=False,
                weight=2,
                popup=f"GPS Failure at {minutes}m {seconds}s<br>{entry.get('status', '')}"
            ).add_to(failures_group)

    failures_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend
    total_points = len(trace)
    valid_points = len(valid_entries)
    failed_points = total_points - valid_points
    if valid_entries:
        total_time = valid_entries[-1].get("elapsed", 0)
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        duration = f"{minutes}m {seconds}s"
    else:
        duration = "unknown"

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 1000;
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid grey;
        font-family: Arial;
        font-size: 12px;
    ">
        <b>GPS Trace</b><br>
        <hr style="margin: 5px 0">
        Duration: {duration}<br>
        Total points: {total_points}<br>
        Valid: {valid_points}<br>
        Failed: {failed_points} ({100*failed_points/total_points:.1f}%)<br>
        <hr style="margin: 5px 0">
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 12px; height: 12px; background: green; border-radius: 50%; margin-right: 5px;"></div>
            Accuracy &lt;10m
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 12px; height: 12px; background: orange; border-radius: 50%; margin-right: 5px;"></div>
            Accuracy 10-20m
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 12px; height: 12px; background: red; border-radius: 50%; margin-right: 5px;"></div>
            Accuracy &gt;20m
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 12px; height: 12px; border: 2px solid red; border-radius: 50%; margin-right: 5px;"></div>
            GPS Failure
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add fullscreen
    plugins.Fullscreen().add_to(m)

    # Save
    m.save(output_path)
    print(f"Trace map saved to {output_path}")
    print(f"  {valid_points} valid points, {failed_points} failures")


def main():
    parser = argparse.ArgumentParser(description="Visualize GPS trace on map")
    parser.add_argument("trace", help="GPS trace JSON file")
    parser.add_argument("-o", "--output", default="trace_map.html",
                        help="Output HTML file (default: trace_map.html)")

    args = parser.parse_args()

    if not Path(args.trace).exists():
        print(f"Trace file not found: {args.trace}")
        return 1

    trace = load_trace(args.trace)
    create_trace_map(trace, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
