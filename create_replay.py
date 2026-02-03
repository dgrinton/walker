#!/usr/bin/env python3
"""
Create a replay data file combining GPS trace, log, and OSM data.

Usage:
    python create_replay.py trace.json walker.log [-o replay_data.json]

This generates a single JSON file containing all data needed for visualization,
including node coordinates from OSM.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from walker import OSMFetcher, StreetGraph, CONFIG


def parse_log_file(log_path: str) -> list[dict]:
    """Parse walker log file into structured entries"""
    entries = []

    with open(log_path) as f:
        lines = f.readlines()

    start_time = None

    for line in lines:
        # Parse: [timestamp] message | {json}
        match = re.match(r'^\[([^\]]+)\]\s*(.+?)(?:\s*\|\s*(.+))?$', line.strip())
        if not match:
            continue

        timestamp_str = match.group(1)
        message = match.group(2).strip()
        json_str = match.group(3)

        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            continue

        if start_time is None:
            start_time = timestamp

        elapsed = (timestamp - start_time).total_seconds()

        data = None
        if json_str:
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                pass

        entries.append({
            "elapsed": elapsed,
            "timestamp": timestamp_str,
            "message": message,
            "data": data
        })

    return entries


def extract_node_ids(log_entries: list[dict]) -> set[int]:
    """Extract all node IDs mentioned in log entries"""
    node_ids = set()

    for entry in log_entries:
        if entry["data"]:
            data = entry["data"]
            for key in ["current_node", "next_node", "from", "to", "expected", "actual", "at_node", "next"]:
                if key in data and data[key]:
                    try:
                        node_ids.add(int(data[key]))
                    except (ValueError, TypeError):
                        pass

    return node_ids


def get_center_from_trace(trace_path: str) -> tuple[float, float]:
    """Get center coordinates from trace file"""
    with open(trace_path) as f:
        data = json.load(f)

    valid_points = [e["location"] for e in data["trace"] if e.get("location")]

    if not valid_points:
        return None, None

    lat = sum(p["lat"] for p in valid_points) / len(valid_points)
    lon = sum(p["lon"] for p in valid_points) / len(valid_points)

    return lat, lon


def create_replay_data(trace_path: str, log_path: str, output_path: str):
    """Create combined replay data file"""

    print("Loading trace file...")
    with open(trace_path) as f:
        trace_data = json.load(f)

    print("Parsing log file...")
    log_entries = parse_log_file(log_path)

    print(f"Found {len(trace_data['trace'])} trace entries, {len(log_entries)} log entries")

    # Extract node IDs from log
    node_ids = extract_node_ids(log_entries)
    print(f"Found {len(node_ids)} unique node IDs in log")

    # Get center from trace
    center_lat, center_lon = get_center_from_trace(trace_path)

    if center_lat is None:
        print("Could not determine center from trace")
        node_locations = {}
    else:
        print(f"Fetching OSM data around ({center_lat:.5f}, {center_lon:.5f})...")
        osm_data = OSMFetcher.fetch_streets(center_lat, center_lon, CONFIG["osm_fetch_radius"])

        # Build graph to get node locations
        graph = StreetGraph()
        graph.build_from_osm(osm_data)

        # Extract locations for nodes mentioned in log
        node_locations = {}
        for node_id in node_ids:
            loc = graph.get_node_location(node_id)
            if loc:
                node_locations[str(node_id)] = {"lat": loc[0], "lon": loc[1]}

        print(f"Found locations for {len(node_locations)}/{len(node_ids)} nodes")

    # Categorize log entries
    state_entries = []
    event_entries = []

    for entry in log_entries:
        if entry["message"] == "STATE" and entry["data"]:
            state_entries.append({
                "elapsed": entry["elapsed"],
                **entry["data"]
            })
        elif any(keyword in entry["message"] for keyword in
                 ["Direction:", "Walked", "Moved", "Recalculated", "Deviation", "Walk started", "Walk complete"]):
            event_entries.append({
                "elapsed": entry["elapsed"],
                "message": entry["message"],
                "data": entry["data"]
            })

    # Build replay data
    replay_data = {
        "created_at": datetime.now().isoformat(),
        "trace": trace_data["trace"],
        "log_entries": log_entries,
        "state_entries": state_entries,
        "event_entries": event_entries,
        "node_locations": node_locations,
        "center": {"lat": center_lat, "lon": center_lon} if center_lat else None,
        "stats": {
            "trace_points": len(trace_data["trace"]),
            "valid_gps_points": sum(1 for e in trace_data["trace"] if e.get("location")),
            "log_entries": len(log_entries),
            "state_entries": len(state_entries),
            "event_entries": len(event_entries),
            "nodes_with_location": len(node_locations)
        }
    }

    # Calculate duration
    if trace_data["trace"]:
        replay_data["duration"] = trace_data["trace"][-1].get("elapsed", 0)

    # Write output
    with open(output_path, "w") as f:
        json.dump(replay_data, f, indent=2)

    print(f"\nReplay data saved to {output_path}")
    print(f"Stats: {replay_data['stats']}")


def main():
    parser = argparse.ArgumentParser(description="Create replay data file")
    parser.add_argument("trace", help="GPS trace JSON file")
    parser.add_argument("log", help="Walker log file")
    parser.add_argument("-o", "--output", default="replay_data.json",
                        help="Output file (default: replay_data.json)")

    args = parser.parse_args()

    if not Path(args.trace).exists():
        print(f"Trace file not found: {args.trace}")
        return 1

    if not Path(args.log).exists():
        print(f"Log file not found: {args.log}")
        return 1

    create_replay_data(args.trace, args.log, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
