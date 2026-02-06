"""Debug GUI server for Walker."""

import asyncio
import http.server
import json
import queue
import socketserver
import threading
import time
import webbrowser
from functools import partial
from typing import Optional

from .models import Location


# HTML template for the debug GUI
DEBUG_GUI_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Walker Debug GUI</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; height: 100vh; display: flex; flex-direction: column; }
        header { background: #1e293b; color: white; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; }
        header h1 { font-size: 18px; font-weight: 600; }
        .status-badge { background: #22c55e; padding: 4px 12px; border-radius: 12px; font-size: 12px; }
        .status-badge.disconnected { background: #ef4444; }
        .main-content { display: flex; flex: 1; overflow: hidden; }
        #map { flex: 1; min-width: 0; }
        .debug-panel { width: 400px; background: #f8fafc; display: flex; flex-direction: column; border-left: 1px solid #e2e8f0; }
        .panel-section { padding: 16px; border-bottom: 1px solid #e2e8f0; }
        .panel-section h2 { font-size: 12px; text-transform: uppercase; color: #64748b; margin-bottom: 12px; letter-spacing: 0.5px; }
        .state-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .state-item { background: white; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0; }
        .state-label { font-size: 11px; color: #64748b; margin-bottom: 4px; }
        .state-value { font-size: 16px; font-weight: 600; color: #1e293b; }
        .logs-section { flex: 1; display: flex; flex-direction: column; min-height: 0; }
        .logs-container { flex: 1; overflow-y: auto; padding: 12px; background: #1e293b; font-family: "SF Mono", Monaco, monospace; font-size: 12px; }
        .log-entry { color: #94a3b8; margin-bottom: 6px; line-height: 1.4; }
        .log-entry .timestamp { color: #64748b; }
        .log-entry .message { color: #e2e8f0; }
        .log-entry .data { color: #38bdf8; }
        .audio-section { background: #fef3c7; padding: 16px; }
        .audio-section h2 { color: #92400e; }
        .audio-text { font-size: 14px; color: #78350f; font-weight: 500; min-height: 20px; }
        .click-hint { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); color: white; padding: 8px 16px; border-radius: 20px; font-size: 13px; z-index: 1000; pointer-events: none; }
        .marker-current { background: #ef4444; border: 3px solid white; border-radius: 50%; width: 16px; height: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.3); }
        .marker-next { background: #f97316; border: 2px solid white; border-radius: 50%; width: 12px; height: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
    </style>
</head>
<body>
    <header>
        <h1>Walker Debug GUI</h1>
        <span id="connection-status" class="status-badge disconnected">Disconnected</span>
    </header>
    <div class="main-content">
        <div id="map">
            <div class="click-hint">Click on map to set GPS location</div>
        </div>
        <div class="debug-panel">
            <div class="panel-section">
                <h2>State</h2>
                <div class="state-grid">
                    <div class="state-item">
                        <div class="state-label">Distance Walked</div>
                        <div class="state-value" id="walked-distance">0 m</div>
                    </div>
                    <div class="state-item">
                        <div class="state-label">Target Distance</div>
                        <div class="state-value" id="target-distance">0 m</div>
                    </div>
                    <div class="state-item">
                        <div class="state-label">Segments Walked</div>
                        <div class="state-value" id="segments-walked">0</div>
                    </div>
                    <div class="state-item">
                        <div class="state-label">GPS Status</div>
                        <div class="state-value" id="gps-status">-</div>
                    </div>
                    <div class="state-item">
                        <div class="state-label">Current Node</div>
                        <div class="state-value" id="current-node">-</div>
                    </div>
                    <div class="state-item">
                        <div class="state-label">Next Node</div>
                        <div class="state-value" id="next-node">-</div>
                    </div>
                </div>
            </div>
            <div class="panel-section audio-section">
                <h2>Audio Prompt</h2>
                <div class="audio-text" id="audio-text">-</div>
            </div>
            <div class="panel-section logs-section">
                <h2>Logs</h2>
                <div class="logs-container" id="logs"></div>
            </div>
        </div>
    </div>
    <script>
        // Initialize map
        var map = L.map('map').setView([40.7580, -73.9855], 15);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);

        // State
        var ws = null;
        var routeLayer = null;
        var currentMarker = null;
        var nextMarker = null;
        var walkedPolyline = null;
        var routeData = null;
        var walkedNodes = [];

        // Custom icons
        var currentIcon = L.divIcon({className: 'marker-current', iconSize: [16, 16], iconAnchor: [8, 8]});
        var nextIcon = L.divIcon({className: 'marker-next', iconSize: [12, 12], iconAnchor: [6, 6]});

        // Connect to WebSocket
        function connect() {
            ws = new WebSocket('ws://localhost:{{WS_PORT}}');

            ws.onopen = function() {
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').classList.remove('disconnected');
                addLog('Connected to walker');
            };

            ws.onclose = function() {
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').classList.add('disconnected');
                addLog('Disconnected from walker');
                setTimeout(connect, 2000);
            };

            ws.onerror = function(err) {
                addLog('WebSocket error');
            };

            ws.onmessage = function(event) {
                var msg = JSON.parse(event.data);
                handleMessage(msg);
            };
        }

        function handleMessage(msg) {
            switch(msg.type) {
                case 'route':
                    displayRoute(msg.data);
                    break;
                case 'state':
                    updateState(msg.data);
                    break;
                case 'log':
                    addLog(msg.data.message, msg.data.data);
                    break;
                case 'audio':
                    showAudio(msg.data.text);
                    break;
            }
        }

        function displayRoute(data) {
            routeData = data;
            walkedNodes = [];

            // Clear existing route
            if (routeLayer) {
                map.removeLayer(routeLayer);
            }

            // Draw route segments
            routeLayer = L.layerGroup().addTo(map);

            if (data.segments && data.segments.length > 0) {
                data.segments.forEach(function(seg) {
                    var color = seg.is_new ? '#22c55e' : '#3b82f6';
                    var line = L.polyline(seg.coords, {
                        color: color,
                        weight: 5,
                        opacity: 0.8
                    }).addTo(routeLayer);
                    line.bindPopup('<b>' + (seg.name || 'unnamed') + '</b><br>' +
                                  seg.length.toFixed(0) + 'm' +
                                  (seg.is_new ? '<br><em>New!</em>' : ''));
                });
            }

            // Fit map to route
            if (data.route && data.route.length > 0) {
                var bounds = L.latLngBounds(data.route);
                map.fitBounds(bounds, {padding: [50, 50]});
            }

            // Add start marker
            if (data.start) {
                L.circleMarker([data.start.lat, data.start.lon], {
                    radius: 10,
                    fillColor: '#ef4444',
                    color: '#ffffff',
                    weight: 3,
                    fillOpacity: 1
                }).addTo(routeLayer).bindPopup('Start');
            }

            addLog('Route received: ' + (data.route ? data.route.length : 0) + ' nodes');
        }

        function updateState(state) {
            document.getElementById('walked-distance').textContent = Math.round(state.walked_distance || 0) + ' m';
            document.getElementById('target-distance').textContent = Math.round(state.target_distance || 0) + ' m';
            document.getElementById('segments-walked').textContent = state.segments_walked || 0;
            document.getElementById('gps-status').textContent = state.gps_status || '-';
            document.getElementById('current-node').textContent = state.current_node || '-';
            document.getElementById('next-node').textContent = state.next_node || '-';

            // Update current position marker
            if (state.location) {
                var pos = [state.location.lat, state.location.lon];
                if (currentMarker) {
                    currentMarker.setLatLng(pos);
                } else {
                    currentMarker = L.marker(pos, {icon: currentIcon}).addTo(map);
                    currentMarker.bindPopup('Current position');
                }
            }

            // Update next waypoint marker
            if (state.next_node_location) {
                var nextPos = [state.next_node_location.lat, state.next_node_location.lon];
                if (nextMarker) {
                    nextMarker.setLatLng(nextPos);
                } else {
                    nextMarker = L.marker(nextPos, {icon: nextIcon}).addTo(map);
                    nextMarker.bindPopup('Next waypoint');
                }
            }

            // Update walked path overlay
            if (state.route_index !== undefined && routeData && routeData.route) {
                var walkedPath = routeData.route.slice(0, state.route_index + 1);
                if (walkedPath.length > 1) {
                    if (walkedPolyline) {
                        walkedPolyline.setLatLngs(walkedPath);
                    } else {
                        walkedPolyline = L.polyline(walkedPath, {
                            color: '#94a3b8',
                            weight: 8,
                            opacity: 0.6
                        }).addTo(map);
                    }
                }
            }
        }

        function showAudio(text) {
            document.getElementById('audio-text').textContent = text;
        }

        function addLog(message, data) {
            var logs = document.getElementById('logs');
            var entry = document.createElement('div');
            entry.className = 'log-entry';

            var timestamp = new Date().toLocaleTimeString();
            var html = '<span class="timestamp">[' + timestamp + ']</span> <span class="message">' + message + '</span>';
            if (data) {
                html += ' <span class="data">' + JSON.stringify(data) + '</span>';
            }
            entry.innerHTML = html;

            logs.appendChild(entry);
            logs.scrollTop = logs.scrollHeight;

            // Keep only last 100 entries
            while (logs.children.length > 100) {
                logs.removeChild(logs.firstChild);
            }
        }

        // Handle map clicks
        map.on('click', function(e) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'location',
                    data: {
                        lat: e.latlng.lat,
                        lon: e.latlng.lng
                    }
                }));
                addLog('Clicked location: ' + e.latlng.lat.toFixed(5) + ', ' + e.latlng.lng.toFixed(5));
            }
        });

        // Start connection
        connect();
    </script>
</body>
</html>'''


class DebugServer:
    """HTTP and WebSocket server for debug GUI"""

    def __init__(self, http_port: int = 8080, ws_port: int = 8765):
        self.http_port = http_port
        self.ws_port = ws_port
        self.websocket = None
        self.location_queue: queue.Queue = queue.Queue()
        self.http_thread = None
        self.ws_thread = None
        self.ws_loop = None
        self.connected_clients: set = set()
        self._running = False

    def start(self):
        """Start HTTP and WebSocket servers in background threads"""
        self._running = True

        # Start HTTP server
        self.http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        self.http_thread.start()

        # Start WebSocket server
        self.ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        self.ws_thread.start()

        # Give servers time to start
        time.sleep(0.5)

        # Open browser
        url = f"http://localhost:{self.http_port}"
        print(f"Debug GUI available at: {url}")
        webbrowser.open(url)

    def _run_http_server(self):
        """Run the HTTP server for serving the GUI"""
        handler = partial(_DebugHTTPHandler, self.ws_port)
        with socketserver.TCPServer(("", self.http_port), handler) as httpd:
            httpd.allow_reuse_address = True
            while self._running:
                httpd.handle_request()

    def _run_ws_server(self):
        """Run the WebSocket server"""
        self.ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.ws_loop)

        async def handler(websocket):
            self.connected_clients.add(websocket)
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if data.get('type') == 'location':
                            loc_data = data.get('data', {})
                            location = Location(
                                lat=loc_data['lat'],
                                lon=loc_data['lon'],
                                accuracy=0,
                                timestamp=time.time()
                            )
                            self.location_queue.put(location)
                    except json.JSONDecodeError:
                        pass
            finally:
                self.connected_clients.discard(websocket)

        async def main():
            try:
                import websockets
                async with websockets.serve(handler, "localhost", self.ws_port):
                    while self._running:
                        await asyncio.sleep(0.1)
            except Exception as e:
                print(f"WebSocket server error: {e}")

        self.ws_loop.run_until_complete(main())

    def _send_message(self, msg_type: str, data: dict):
        """Send a message to all connected WebSocket clients"""
        if not self.connected_clients or not self.ws_loop:
            return

        message = json.dumps({"type": msg_type, "data": data})

        async def send_to_all():
            for client in list(self.connected_clients):
                try:
                    await client.send(message)
                except Exception:
                    self.connected_clients.discard(client)

        try:
            asyncio.run_coroutine_threadsafe(send_to_all(), self.ws_loop)
        except Exception:
            pass

    def send_route(self, route_data: dict):
        """Send route to browser for display"""
        self._send_message("route", route_data)

    def send_state(self, state: dict):
        """Send state update to browser"""
        self._send_message("state", state)

    def send_log(self, message: str, data: Optional[dict] = None):
        """Send log message to browser"""
        self._send_message("log", {"message": message, "data": data})

    def send_audio(self, text: str):
        """Send audio prompt text to browser"""
        self._send_message("audio", {"text": text})

    def get_clicked_location(self, timeout: float = 30) -> Optional[Location]:
        """Block until user clicks on map, return Location"""
        try:
            return self.location_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the servers"""
        self._running = False


class _DebugHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves the debug GUI"""

    def __init__(self, ws_port: int, *args, **kwargs):
        self.ws_port = ws_port
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = DEBUG_GUI_HTML.replace('{{WS_PORT}}', str(self.ws_port))
            self.wfile.write(html.encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress HTTP log messages


class WebSocketGPS:
    """GPS source that gets locations from map clicks via WebSocket"""

    def __init__(self, debug_server: DebugServer):
        self.server = debug_server
        self.last_location: Optional[Location] = None
        self.consecutive_failures = 0

    def get_location(self, timeout: int = 30) -> Optional[Location]:
        """Block until location clicked on map"""
        location = self.server.get_clicked_location(timeout=timeout)
        if location:
            self.last_location = location
            self.consecutive_failures = 0
            return location
        else:
            self.consecutive_failures += 1
            return None

    def get_status(self) -> str:
        return "Debug GUI (click map to set location)"
