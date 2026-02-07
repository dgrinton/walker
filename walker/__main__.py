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
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .app import Walker
from .gps import GPSRecorder, GPSPlayback
from .debug_gui import WebSocketGPS


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

    args = parser.parse_args()

    # Validate lat/lon - must provide both or neither
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be used together")

    # Validate debug-gui requires lat/lon
    if args.debug_gui and (args.lat is None or args.lon is None):
        parser.error("--debug-gui requires --lat and --lon")

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
