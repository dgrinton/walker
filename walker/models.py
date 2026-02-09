"""Data classes for Walker."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Location:
    lat: float
    lon: float
    accuracy: Optional[float] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Location":
        return cls(**d)


@dataclass
class Segment:
    """A street segment between two intersections"""
    id: str  # "{node1_id}-{node2_id}" sorted
    node1: int
    node2: int
    way_id: int
    name: Optional[str]
    road_type: str
    length: float  # meters
    busy_road_adjacent: bool = False
    busy_road_crossing: bool = False

    @classmethod
    def make_id(cls, node1: int, node2: int) -> str:
        return f"{min(node1, node2)}-{max(node1, node2)}"


@dataclass
class DirectedSegment:
    """A segment with direction information (which way we walked it)"""
    segment: Segment
    from_node: int
    to_node: int


@dataclass
class RecentPathContext:
    """Context about recently walked segments for backtrack detection"""
    recent_segments: list[DirectedSegment]
    recent_street_names: set[str]

    @classmethod
    def empty(cls) -> "RecentPathContext":
        return cls(recent_segments=[], recent_street_names=set())

    def add_segment(self, segment: Segment, from_node: int, to_node: int, max_history: int):
        """Add a segment to recent history"""
        directed = DirectedSegment(segment=segment, from_node=from_node, to_node=to_node)
        self.recent_segments.append(directed)
        if segment.name:
            self.recent_street_names.add(segment.name)
        # Trim to max history
        while len(self.recent_segments) > max_history:
            removed = self.recent_segments.pop(0)
            # Rebuild street names set from remaining segments
            self.recent_street_names = {
                s.segment.name for s in self.recent_segments if s.segment.name
            }
