"""Walker - Turn-by-turn pedestrian exploration guide."""

from .config import CONFIG
from .models import Location, Segment, DirectedSegment, RecentPathContext
from .logger import Logger
from .gps import GPS, GPSRecorder, GPSPlayback
from .debug_gui import DebugServer, WebSocketGPS
from .geo import (
    haversine_distance,
    bearing_between,
    bearing_to_compass,
    relative_direction,
    is_parallel_corridor,
    retry_with_backoff,
    point_in_polygon,
    point_in_any_polygon,
)
from .osm import OSMFetcher
from .graph import StreetGraph
from .history import HistoryDB
from .audio import Audio
from .planner import RoutePlanner
from .zone_editor import ZoneEditorServer
from .app import Walker
from .__main__ import main

__all__ = [
    "CONFIG",
    "Location",
    "Segment",
    "DirectedSegment",
    "RecentPathContext",
    "Logger",
    "GPS",
    "GPSRecorder",
    "GPSPlayback",
    "DebugServer",
    "WebSocketGPS",
    "haversine_distance",
    "bearing_between",
    "bearing_to_compass",
    "relative_direction",
    "is_parallel_corridor",
    "retry_with_backoff",
    "point_in_polygon",
    "point_in_any_polygon",
    "OSMFetcher",
    "StreetGraph",
    "HistoryDB",
    "Audio",
    "RoutePlanner",
    "ZoneEditorServer",
    "Walker",
    "main",
]
