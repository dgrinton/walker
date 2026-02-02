#!/usr/bin/env python3
"""
Walker Simulator - Test the routing logic without GPS/Termux

This simulates walking by clicking on a map or using keyboard input.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from walker import (
    StreetGraph, RoutePlanner, HistoryDB, OSMFetcher,
    haversine_distance, bearing_between, bearing_to_compass,
    CONFIG
)


class WalkerSimulator:
    """Simulate walking for testing"""

    def __init__(self):
        self.history = HistoryDB("walker_sim_history.db")
        self.graph: StreetGraph = None
        self.planner: RoutePlanner = None
        self.current_node: int = None
        self.previous_node: int = None

    def load_area(self, lat: float, lon: float):
        """Load map data for an area"""
        print(f"Fetching map data around ({lat}, {lon})...")

        osm_data = OSMFetcher.fetch_streets(lat, lon, CONFIG["osm_fetch_radius"])

        if not osm_data.get("elements"):
            print("Failed to fetch map data")
            return False

        self.graph = StreetGraph()
        self.graph.build_from_osm(osm_data)

        # Find starting node
        self.current_node = self.graph.find_nearest_node(lat, lon)
        if not self.current_node:
            print("Could not find starting point")
            return False

        loc = self.graph.get_node_location(self.current_node)
        print(f"Starting at node {self.current_node} ({loc[0]:.5f}, {loc[1]:.5f})")

        # Initialize planner
        self.planner = RoutePlanner(self.graph, self.history)
        self.planner.start_walk(self.current_node, 2000)  # 2km default

        return True

    def show_options(self):
        """Show available directions from current node"""
        neighbors = self.graph.get_neighbors(self.current_node)
        current_loc = self.graph.get_node_location(self.current_node)

        print(f"\nAt node {self.current_node} ({current_loc[0]:.6f}, {current_loc[1]:.6f})")
        print(f"Distance walked: {self.planner.walked_distance:.0f}m")
        print(f"This is {'an intersection' if self.graph.is_intersection(self.current_node) else 'not an intersection'}")
        print(f"\nOptions ({len(neighbors)} neighbors):")

        options = []
        for i, neighbor in enumerate(neighbors):
            nloc = self.graph.get_node_location(neighbor)
            segment = self.graph.get_segment(self.current_node, neighbor)
            bearing = bearing_between(current_loc[0], current_loc[1], nloc[0], nloc[1])
            compass = bearing_to_compass(bearing)

            times_walked, last_walked = self.history.get_segment_history(segment.id)
            score = self.planner.score_edge(self.current_node, neighbor)

            name = segment.name or "unnamed"
            status = "NEW" if times_walked == 0 else f"walked {times_walked}x"

            # Calculate relative direction if we came from somewhere
            if self.previous_node:
                prev_loc = self.graph.get_node_location(self.previous_node)
                incoming_bearing = bearing_between(prev_loc[0], prev_loc[1], current_loc[0], current_loc[1])
                from walker import relative_direction
                rel = relative_direction(incoming_bearing, bearing)
            else:
                rel = compass

            options.append(neighbor)
            print(f"  [{i+1}] {rel:12} -> {name:20} ({segment.road_type}, {segment.length:.0f}m, {status}, score={score:.1f})")

        # Show recommended
        recommended = self.planner.choose_next_direction(self.current_node, self.previous_node)
        if recommended:
            rec_idx = options.index(recommended) + 1
            print(f"\nRecommended: [{rec_idx}]")

        return options

    def move_to(self, node: int):
        """Move to a neighboring node"""
        segment = self.graph.get_segment(self.current_node, node)
        if segment:
            self.history.record_segment(segment.id)
            self.planner.walked_distance += segment.length
            print(f"Walked {segment.length:.0f}m on {segment.name or 'unnamed street'}")

        self.previous_node = self.current_node
        self.current_node = node

    def run_interactive(self):
        """Run interactive simulation"""
        print("\n=== Walker Simulator ===")
        print("Commands: number to choose direction, 'q' to quit, 'r' for recommended")
        print()

        while True:
            options = self.show_options()

            try:
                cmd = input("\nChoice: ").strip().lower()
            except EOFError:
                break

            if cmd == 'q':
                break
            elif cmd == 'r':
                recommended = self.planner.choose_next_direction(self.current_node, self.previous_node)
                if recommended:
                    self.move_to(recommended)
            elif cmd.isdigit():
                idx = int(cmd) - 1
                if 0 <= idx < len(options):
                    self.move_to(options[idx])
                else:
                    print("Invalid choice")
            else:
                print("Unknown command")

        print(f"\nFinal distance: {self.planner.walked_distance:.0f}m")
        self.history.close()


def main():
    # Default: somewhere in London (change to your area)
    if len(sys.argv) >= 3:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
    else:
        # Default: Central Park, NYC
        lat = 40.7829
        lon = -73.9654
        print(f"Usage: {sys.argv[0]} <lat> <lon>")
        print(f"Using default location: Central Park, NYC ({lat}, {lon})")

    sim = WalkerSimulator()
    if sim.load_area(lat, lon):
        sim.run_interactive()


if __name__ == "__main__":
    main()
