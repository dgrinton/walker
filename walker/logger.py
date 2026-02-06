"""Logging module for Walker."""

import json
from datetime import datetime
from typing import Optional, Callable


class Logger:
    """Logs state to file"""

    def __init__(self, log_path: Optional[str] = None, callback: Optional[Callable] = None):
        self.log_path = log_path
        self.callback = callback
        self.file = None
        if log_path:
            self.file = open(log_path, "a")
            self._write_header()

    def _write_header(self):
        if self.file:
            self.file.write(f"\n{'='*60}\n")
            self.file.write(f"Walker Log - {datetime.now().isoformat()}\n")
            self.file.write(f"{'='*60}\n\n")
            self.file.flush()

    def log(self, message: str, data: Optional[dict] = None):
        """Log a message with optional structured data"""
        timestamp = datetime.now().isoformat()
        line = f"[{timestamp}] {message}"
        if data:
            line += f" | {json.dumps(data)}"
        print(line)
        if self.file:
            self.file.write(line + "\n")
            self.file.flush()
        if self.callback:
            self.callback(message, data)

    def close(self):
        if self.file:
            self.file.close()
