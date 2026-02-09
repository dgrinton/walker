"""OpenStreetMap data fetching via Overpass API with disk caching."""

import hashlib
import json
import os
import time

import requests

from .geo import haversine_distance


class OSMFetcher:
    """Fetch street data from OpenStreetMap via Overpass API"""

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    CACHE_DIR = "osm_cache"
    CACHE_MAX_AGE = 7 * 24 * 3600  # 7 days

    @classmethod
    def _cache_path(cls, lat: float, lon: float, radius: float) -> str:
        """Generate a cache file path for the given query parameters."""
        # Round coordinates to reduce near-duplicate caches
        key = f"{lat:.5f},{lon:.5f},{radius:.0f}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(cls.CACHE_DIR, f"osm_{h}.json")

    @classmethod
    def _find_covering_cache(cls, lat: float, lon: float, radius: float) -> dict | None:
        """Find a cached response that covers the requested area.

        A cache entry covers the request if the cached center is close enough
        and the cached radius is large enough that the requested circle fits
        inside the cached circle.
        """
        if not os.path.isdir(cls.CACHE_DIR):
            return None
        now = time.time()
        for fname in os.listdir(cls.CACHE_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(cls.CACHE_DIR, fname)
            try:
                age = now - os.path.getmtime(fpath)
                if age > cls.CACHE_MAX_AGE:
                    continue
                with open(fpath) as f:
                    cached = json.load(f)
                meta = cached.get("_cache_meta")
                if not meta:
                    continue
                clat, clon, cradius = meta["lat"], meta["lon"], meta["radius"]
                # Distance between cached center and requested center
                dist = haversine_distance(lat, lon, clat, clon)
                # The cached circle covers the request if:
                # cached_radius >= dist_between_centers + requested_radius
                if cradius >= dist + radius:
                    print(f"Using cached OSM data ({cradius:.0f}m radius from {age/3600:.1f}h ago)")
                    return cached
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return None

    @classmethod
    def fetch_streets(cls, lat: float, lon: float, radius: float) -> dict:
        """Fetch all walkable streets within radius of location.

        Uses disk cache to avoid repeated Overpass API requests. A cached
        response is reused if it covers the requested area (same or larger
        radius from a nearby center point).
        """
        # Check cache first
        cached = cls._find_covering_cache(lat, lon, radius)
        if cached:
            # Strip cache metadata before returning
            result = {k: v for k, v in cached.items() if k != "_cache_meta"}
            return result

        # Overpass query for walkable ways
        # Scale timeout with radius — larger areas need more server time
        timeout = max(30, int(radius / 50))
        query = f"""
        [out:json][timeout:{timeout}];
        (
          way["highway"~"^(footway|pedestrian|path|residential|living_street|service|unclassified|tertiary|secondary|primary|trunk)$"]
            (around:{radius},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """

        print(f"Fetching OSM data around ({lat:.5f}, {lon:.5f}), radius {radius:.0f}m...")

        try:
            response = requests.post(cls.OVERPASS_URL, data={"data": query}, timeout=timeout + 30)
            response.raise_for_status()
            data = response.json()

            # Cache the response with metadata
            if data.get("elements"):
                os.makedirs(cls.CACHE_DIR, exist_ok=True)
                cache_data = dict(data)
                cache_data["_cache_meta"] = {
                    "lat": lat, "lon": lon, "radius": radius,
                    "fetched_at": time.time(),
                }
                cache_path = cls._cache_path(lat, lon, radius)
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f)
                print(f"Cached OSM data to {cache_path}")

            return data
        except requests.RequestException as e:
            print(f"OSM fetch error: {e}")
            return {"elements": []}
