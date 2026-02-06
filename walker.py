#!/usr/bin/env python3
"""
Walker - Turn-by-turn pedestrian exploration guide

This is a compatibility wrapper. The actual implementation is in the walker/ package.

Usage:
    python walker.py [distance_km] [options]

Or run as a module:
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

from walker import main

if __name__ == "__main__":
    main()
