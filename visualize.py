#!/usr/bin/env python3
"""
Visualize walking history on an interactive map.

Usage:
    python visualize.py [lat] [lon] [--db PATH] [--output PATH]

Examples:
    python visualize.py 51.5074 -0.1278
    python visualize.py 51.5074 -0.1278 --db walker_history.db --output my_walks.html
"""

import argparse
import sqlite3
from pathlib import Path

import folium
from folium import plugins

from walker import StreetGraph, OSMFetcher, CONFIG


def get_walked_segments(db_path: str) -> dict[str, tuple[int, str]]:
    """Get all walked segments from database.

    Returns: {segment_id: (times_walked, last_walked)}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT segment_id, times_walked, last_walked FROM segment_history"
    )
    result = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    conn.close()
    return result


def get_walk_stats(db_path: str) -> dict:
    """Get overall statistics from database."""
    conn = sqlite3.connect(db_path)

    cursor = conn.execute(
        "SELECT COUNT(*), SUM(times_walked) FROM segment_history"
    )
    row = cursor.fetchone()
    unique_segments = row[0] or 0
    total_walks = row[1] or 0

    cursor = conn.execute(
        "SELECT COUNT(*), SUM(distance_meters) FROM walks WHERE ended_at IS NOT NULL"
    )
    row = cursor.fetchone()
    completed_walks = row[0] or 0
    total_distance = row[1] or 0

    conn.close()

    return {
        "unique_segments": unique_segments,
        "total_segment_walks": total_walks,
        "completed_walks": completed_walks,
        "total_distance_km": total_distance / 1000
    }


def frequency_to_color(times_walked: int, max_walked: int) -> str:
    """Convert walk frequency to color (green -> yellow -> red)."""
    if max_walked <= 1:
        return "#00ff00"  # green

    # Normalize to 0-1
    ratio = min(times_walked / max_walked, 1.0)

    if ratio < 0.5:
        # Green to yellow
        r = int(255 * (ratio * 2))
        g = 255
    else:
        # Yellow to red
        r = 255
        g = int(255 * (1 - (ratio - 0.5) * 2))

    return f"#{r:02x}{g:02x}00"


def create_map(
    lat: float,
    lon: float,
    db_path: str,
    show_unwalked: bool = True,
    radius: float = None
) -> folium.Map:
    """Create an interactive map with walking history overlay."""

    radius = radius or CONFIG["osm_fetch_radius"]

    # Fetch OSM data
    print(f"Fetching map data around ({lat}, {lon})...")
    osm_data = OSMFetcher.fetch_streets(lat, lon, radius)

    if not osm_data.get("elements"):
        raise ValueError("Could not fetch map data")

    # Build street graph
    graph = StreetGraph()
    graph.build_from_osm(osm_data)

    # Get walked segments from database
    walked = get_walked_segments(db_path)
    stats = get_walk_stats(db_path)

    print(f"Found {len(walked)} walked segments in database")

    # Find max walk count for color scaling
    max_walked = max((v[0] for v in walked.values()), default=1)

    # Create map centered on location
    m = folium.Map(
        location=[lat, lon],
        zoom_start=16,
        tiles="CartoDB positron"
    )

    # Add tile layer options
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Mode").add_to(m)

    # Create feature groups for layers
    walked_layer = folium.FeatureGroup(name="Walked segments", show=True)
    unwalked_layer = folium.FeatureGroup(name="Unwalked segments", show=show_unwalked)

    # Add segments to map
    walked_count = 0
    unwalked_count = 0

    for segment_id, segment in graph.segments.items():
        node1_loc = graph.get_node_location(segment.node1)
        node2_loc = graph.get_node_location(segment.node2)

        if not node1_loc or not node2_loc:
            continue

        coords = [[node1_loc[0], node1_loc[1]], [node2_loc[0], node2_loc[1]]]

        if segment_id in walked:
            times, last = walked[segment_id]
            color = frequency_to_color(times, max_walked)

            popup_text = f"""
                <b>{segment.name or 'Unnamed'}</b><br>
                Type: {segment.road_type}<br>
                Length: {segment.length:.0f}m<br>
                <hr>
                Times walked: <b>{times}</b><br>
                Last walked: {last[:10] if last else 'Unknown'}
            """

            folium.PolyLine(
                coords,
                weight=5,
                color=color,
                opacity=0.8,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(walked_layer)
            walked_count += 1
        else:
            # Unwalked segment
            popup_text = f"""
                <b>{segment.name or 'Unnamed'}</b><br>
                Type: {segment.road_type}<br>
                Length: {segment.length:.0f}m<br>
                <i>Not yet walked</i>
            """

            folium.PolyLine(
                coords,
                weight=2,
                color="#888888",
                opacity=0.4,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(unwalked_layer)
            unwalked_count += 1

    walked_layer.add_to(m)
    unwalked_layer.add_to(m)

    # Add starting point marker
    folium.Marker(
        [lat, lon],
        popup="Starting point",
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend
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
        <b>Walker History</b><br>
        <hr style="margin: 5px 0">
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 30px; height: 4px; background: #00ff00; margin-right: 5px;"></div>
            Walked once
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 30px; height: 4px; background: #ffff00; margin-right: 5px;"></div>
            Walked few times
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 30px; height: 4px; background: #ff0000; margin-right: 5px;"></div>
            Walked many times
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <div style="width: 30px; height: 4px; background: #888888; margin-right: 5px;"></div>
            Not walked
        </div>
        <hr style="margin: 5px 0">
        <b>Stats:</b><br>
        Unique segments: {stats['unique_segments']}<br>
        Coverage: {walked_count}/{walked_count + unwalked_count} ({100*walked_count/(walked_count+unwalked_count):.1f}%)<br>
        Total distance: {stats['total_distance_km']:.1f} km
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add fullscreen button
    plugins.Fullscreen().add_to(m)

    # Add locate control (find user on map)
    plugins.LocateControl().add_to(m)

    print(f"Map created: {walked_count} walked, {unwalked_count} unwalked segments")

    return m


def main():
    parser = argparse.ArgumentParser(
        description="Visualize walking history on a map"
    )
    parser.add_argument("lat", type=float, nargs="?", default=51.5074,
                        help="Latitude (default: 51.5074)")
    parser.add_argument("lon", type=float, nargs="?", default=-0.1278,
                        help="Longitude (default: -0.1278)")
    parser.add_argument("--db", default="walker_history.db",
                        help="Database path (default: walker_history.db)")
    parser.add_argument("--output", "-o", default="walker_map.html",
                        help="Output HTML file (default: walker_map.html)")
    parser.add_argument("--radius", type=float, default=None,
                        help="Radius in meters to fetch (default: 1500)")
    parser.add_argument("--hide-unwalked", action="store_true",
                        help="Hide unwalked segments by default")

    args = parser.parse_args()

    # Check if database exists
    if not Path(args.db).exists():
        print(f"Database not found: {args.db}")
        print("Run walker.py or walker_sim.py first to create walking history.")
        return 1

    try:
        m = create_map(
            args.lat,
            args.lon,
            args.db,
            show_unwalked=not args.hide_unwalked,
            radius=args.radius
        )

        m.save(args.output)
        print(f"\nMap saved to: {args.output}")
        print(f"Open in browser: file://{Path(args.output).absolute()}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
