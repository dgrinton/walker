# Walker

Turn-by-turn pedestrian exploration guide that plans walking routes emphasizing novel, unexplored streets. Uses GPS, audio prompts, and optional web-based visualization to guide you on exploratory walks.

A SQLite database tracks walked segments so future routes always prioritize streets you haven't visited yet.

Designed primarily for [Termux](https://termux.dev/) on Android — pair with Bluetooth headphones for hands-free audio directions.

## Quick Start

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with default 2km walk
python -m walker

# Specify distance
python -m walker 3.5
```

## Usage

```
python -m walker [distance_km] [options]
```

| Option | Description |
|---|---|
| `--preview` | Show the planned route without walking it |
| `--execute` | Virtually walk the route and record segments to the database |
| `--html FILE` | Export route as an HTML map (use with `--preview`) |
| `--gpx FILE` | Export route as GPX for navigation apps (use with `--preview`) |
| `--lat LAT --lon LON` | Set a starting location without GPS |
| `--debug-gui` | Launch a browser-based map with real-time state (requires `--lat`/`--lon`) |
| `--record FILE` | Record GPS trace to JSON for later replay |
| `--playback FILE` | Replay a recorded GPS trace |
| `--speed FACTOR` | Playback speed multiplier (default: 1.0) |
| `--edit-zones` | Open the exclusion zone editor |

### Examples

```bash
# Preview a route and export as HTML map
python -m walker 2 --preview --html route.html

# Export GPX for use in another navigation app
python -m walker 2 --preview --gpx route.gpx

# Test a route without GPS — virtually walk and record to history
python -m walker 2 --execute

# Debug with the browser-based GUI
python -m walker 2 --lat 40.7128 --lon -74.0060 --debug-gui

# Record a walk for later replay
python -m walker 2 --record trace.json
python -m walker 2 --playback trace.json --speed 2.0
```

## How It Works

1. Gets your GPS location (or uses a provided lat/lon)
2. Fetches walkable streets from OpenStreetMap via the Overpass API
3. Builds a street graph and plans a loop route that maximizes novelty
4. Guides you turn-by-turn with audio announcements
5. Records walked segments to a local SQLite database

The route planner scores edges by novelty (how many times you've walked them), road type preference, and backtracking penalties — so you're always discovering new streets.

## Termux Setup

See [SETUP.md](SETUP.md) for Android/Termux installation instructions.

## Dependencies

- [networkx](https://networkx.org/) — graph data structure and routing algorithms
- [requests](https://docs.python-requests.org/) — Overpass API requests
- [folium](https://python-visualization.github.io/folium/) — HTML map generation
- [websockets](https://websockets.readthedocs.io/) — debug GUI real-time communication
