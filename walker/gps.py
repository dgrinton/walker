"""GPS access and recording/playback."""

import json
import subprocess
import time
from datetime import datetime
from typing import Optional

from .config import CONFIG
from .models import Location


class GPS:
    """GPS access via Termux API"""

    def __init__(self):
        self.last_location: Optional[Location] = None
        self.consecutive_failures = 0

    def get_location(self, timeout: int = 30) -> Optional[Location]:
        """Get current location using termux-location"""
        try:
            result = subprocess.run(
                ["termux-location", "-p", "gps", "-r", "once"],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                self.consecutive_failures += 1
                error_msg = result.stderr.strip() if result.stderr else "unknown error"
                return None

            if not result.stdout or not result.stdout.strip():
                self.consecutive_failures += 1
                return None

            data = json.loads(result.stdout)
            location = Location(
                lat=data["latitude"],
                lon=data["longitude"],
                accuracy=data.get("accuracy"),
                timestamp=time.time()
            )
            self.last_location = location
            self.consecutive_failures = 0
            return location

        except subprocess.TimeoutExpired:
            self.consecutive_failures += 1
            return None
        except (json.JSONDecodeError, KeyError) as e:
            self.consecutive_failures += 1
            return None
        except FileNotFoundError:
            self.consecutive_failures += 1
            return None

    def get_status(self) -> str:
        """Get GPS status string"""
        if self.consecutive_failures == 0:
            acc = f", accuracy {self.last_location.accuracy:.0f}m" if self.last_location and self.last_location.accuracy else ""
            return f"GPS OK{acc}"
        else:
            return f"GPS: {self.consecutive_failures} consecutive failures"


class GPSRecorder:
    """Records GPS trace to file"""

    def __init__(self, gps: GPS, record_path: str):
        self.gps = gps
        self.record_path = record_path
        self.trace: list[dict] = []
        self.start_time = time.time()

    def get_location(self, timeout: int = 30) -> Optional[Location]:
        """Get location and record it"""
        location = self.gps.get_location(timeout)

        # Record even failed attempts
        entry = {
            "elapsed": time.time() - self.start_time,
            "timestamp": time.time(),
            "location": location.to_dict() if location else None,
            "status": self.gps.get_status()
        }
        self.trace.append(entry)

        return location

    def get_status(self) -> str:
        return self.gps.get_status()

    def save(self):
        """Save trace to file"""
        with open(self.record_path, "w") as f:
            json.dump({
                "recorded_at": datetime.now().isoformat(),
                "trace": self.trace
            }, f, indent=2)
        print(f"GPS trace saved to {self.record_path} ({len(self.trace)} entries)")


class GPSPlayback:
    """Plays back GPS trace from file"""

    def __init__(self, playback_path: str, speed: float = 1.0):
        self.playback_path = playback_path
        self.speed = speed
        self.trace: list[dict] = []
        self.index = 0
        self.last_location: Optional[Location] = None
        self.consecutive_failures = 0

        # Load trace
        with open(playback_path) as f:
            data = json.load(f)
            self.trace = data["trace"]
        print(f"Loaded GPS trace from {playback_path} ({len(self.trace)} entries)")

    def get_location(self, timeout: int = 30) -> Optional[Location]:
        """Get next location from trace sequentially"""
        if self.index >= len(self.trace):
            return None

        # Return entries one at a time (speed is handled by main loop sleep)
        entry = self.trace[self.index]
        self.index += 1

        if entry["location"]:
            location = Location.from_dict(entry["location"])
            self.last_location = location
            self.consecutive_failures = 0
            return location
        else:
            self.consecutive_failures += 1
            return None

    def get_poll_interval(self) -> float:
        """Get the interval to wait between polls based on trace timing and speed"""
        if self.index <= 0 or self.index >= len(self.trace):
            return CONFIG["gps_poll_interval"] / self.speed

        # Calculate time delta between current and previous entry
        prev_elapsed = self.trace[self.index - 1].get("elapsed", 0)
        curr_elapsed = self.trace[self.index].get("elapsed", 0)
        delta = curr_elapsed - prev_elapsed

        # Apply speed multiplier and clamp to reasonable range
        interval = delta / self.speed
        return max(0.1, min(interval, 5.0))

    def is_finished(self) -> bool:
        """Check if playback is complete"""
        return self.index >= len(self.trace)

    def get_status(self) -> str:
        progress = f"{self.index}/{len(self.trace)}"
        if self.consecutive_failures == 0:
            return f"Playback OK ({progress})"
        else:
            return f"Playback: {self.consecutive_failures} failures ({progress})"
