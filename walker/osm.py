"""OpenStreetMap data fetching via Overpass API."""

import requests


class OSMFetcher:
    """Fetch street data from OpenStreetMap via Overpass API"""

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    @classmethod
    def fetch_streets(cls, lat: float, lon: float, radius: float) -> dict:
        """Fetch all walkable streets within radius of location"""

        # Overpass query for walkable ways
        query = f"""
        [out:json][timeout:30];
        (
          way["highway"~"^(footway|pedestrian|path|residential|living_street|service|unclassified|tertiary|secondary|primary)$"]
            (around:{radius},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """

        print(f"Fetching OSM data around ({lat:.5f}, {lon:.5f})...")

        try:
            response = requests.post(cls.OVERPASS_URL, data={"data": query}, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"OSM fetch error: {e}")
            return {"elements": []}
