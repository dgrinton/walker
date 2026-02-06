# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Walker is a turn-by-turn pedestrian exploration guide that plans walking routes emphasizing novel, unexplored streets. Designed primarily for Termux (Android), it uses GPS, audio prompts (espeak/pyttsx3), and optional web-based visualization to guide users on exploratory walks. A SQLite database tracks walked segments so future routes prioritize new streets.

## Running

A virtual environment is already set up at `venv/`. Activate it before running commands:

```bash
source venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt

# Run with default 2km walk
python -m walker
python walker.py          # compatibility wrapper

# Common options
python -m walker 3.5                          # 3.5km walk
python -m walker 2 --preview                  # show route without walking
python -m walker 2 --preview --html route.html  # generate HTML map visualization
python -m walker 2 --preview --gpx route.gpx    # export GPX for navigation apps
python -m walker 2 --execute                  # virtual walk, records to DB (no GPS)
python -m walker 2 --lat 40.7128 --lon -74.0060 --debug-gui  # browser-based debug GUI

# Recording/playback for debugging
python -m walker 2 --record trace.json
python -m walker 2 --playback trace.json --speed 2.0
```

There is no formal test suite. Use `--preview` and `--execute` modes for manual testing. The `--execute` flag virtually walks the route and records segments to the database without GPS.

## Architecture

The `walker/` package was recently refactored from a monolithic `walker.py` into separate modules. The top-level `walker.py` is a thin compatibility wrapper.

### Core flow

```
CLI (__main__.py) → Walker (app.py) → OSMFetcher → StreetGraph → RoutePlanner → walk loop
```

1. **Walker** (`app.py`) — Main orchestrator. Handles initialization (GPS fix → OSM fetch → graph build → route plan), the walk loop, preview/execute modes, and HTML/GPX export.
2. **RoutePlanner** (`planner.py`) — Greedy route algorithm that scores edges by novelty (times walked), road type weight, and backtrack penalties. Ensures the route can always return to start within the distance budget.
3. **StreetGraph** (`graph.py`) — NetworkX graph wrapper. Nodes = intersections, edges = street segments. Built from OSM data.
4. **OSMFetcher** (`osm.py`) — Queries the Overpass API for walkable roads within a radius.
5. **HistoryDB** (`history.py`) — SQLite database (`walker_history.db`) tracking per-segment walk counts and per-walk session stats. Used by the planner to prioritize novel streets.

### Supporting modules

- **gps.py** — Swappable GPS implementations: real GPS (termux-location), `GPSRecorder`, `GPSPlayback`, `WebSocketGPS` (browser click input).
- **geo.py** — Geographic math (haversine distance, bearing, relative direction) and `retry_with_backoff` utility.
- **audio.py** — TTS with fallback chain: espeak → pyttsx3 → print.
- **models.py** — Dataclasses: `Location`, `Segment`, `DirectedSegment`, `RecentPathContext`.
- **config.py** — All tunable constants (GPS poll interval, road weights, backtrack penalties, intersection radius, etc.).
- **debug_gui.py** — HTTP + WebSocket server hosting a Leaflet.js map with real-time state updates and click-based location input.
- **logger.py** — File logger with optional callback for debug GUI integration.

### Key design patterns

- **Swappable GPS sources**: GPS, GPSRecorder, GPSPlayback, and WebSocketGPS all share an interface; set via `Walker.set_gps_source()`.
- **Novelty-driven routing**: The planner's `score_edge()` balances road type preference, walk history, and anti-backtracking penalties (same-street, parallel-segment detection).
- **Config externalization**: All magic numbers live in `config.py`'s `CONFIG` dict.
- **Silent fallback**: Audio, GPS, and API calls degrade gracefully with retries and fallbacks.

## Dependencies

- `networkx` — Graph data structure and algorithms for route planning
- `requests` — Overpass API HTTP requests
- `folium` — HTML map visualization generation
- `websockets` — Debug GUI real-time communication
