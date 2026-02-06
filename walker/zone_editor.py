"""Zone editor server for managing exclusion zones."""

import asyncio
import http.server
import json
import socketserver
import threading
import time
import webbrowser
from functools import partial
from typing import Optional

from .history import HistoryDB


ZONE_EDITOR_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Walker - Exclusion Zone Editor</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; height: 100vh; display: flex; flex-direction: column; }
        header { background: #1e293b; color: white; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; }
        header h1 { font-size: 18px; font-weight: 600; }
        .status-badge { background: #22c55e; padding: 4px 12px; border-radius: 12px; font-size: 12px; }
        .status-badge.disconnected { background: #ef4444; }
        .main-content { display: flex; flex: 1; overflow: hidden; }
        #map { flex: 1; min-width: 0; }
        .sidebar { width: 320px; background: #f8fafc; display: flex; flex-direction: column; border-left: 1px solid #e2e8f0; }
        .sidebar-header { padding: 16px; border-bottom: 1px solid #e2e8f0; }
        .sidebar-header h2 { font-size: 14px; text-transform: uppercase; color: #64748b; letter-spacing: 0.5px; }
        .zone-list { flex: 1; overflow-y: auto; padding: 8px; }
        .zone-item { background: white; padding: 12px; margin: 6px 0; border-radius: 6px; border: 1px solid #e2e8f0; display: flex; align-items: center; gap: 10px; cursor: pointer; transition: background 0.15s; }
        .zone-item:hover { background: #f1f5f9; }
        .zone-color { width: 16px; height: 16px; border-radius: 3px; flex-shrink: 0; }
        .zone-info { flex: 1; min-width: 0; }
        .zone-name { font-size: 14px; font-weight: 500; color: #1e293b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .zone-meta { font-size: 11px; color: #94a3b8; margin-top: 2px; }
        .zone-actions { display: flex; gap: 4px; }
        .zone-btn { background: none; border: 1px solid #e2e8f0; border-radius: 4px; padding: 4px 8px; cursor: pointer; font-size: 12px; color: #64748b; transition: all 0.15s; }
        .zone-btn:hover { background: #f1f5f9; color: #1e293b; }
        .zone-btn.delete:hover { background: #fef2f2; color: #ef4444; border-color: #fecaca; }
        .empty-state { text-align: center; padding: 40px 20px; color: #94a3b8; }
        .empty-state p { margin-top: 8px; font-size: 13px; }
    </style>
</head>
<body>
    <header>
        <h1>Exclusion Zone Editor</h1>
        <span id="connection-status" class="status-badge disconnected">Disconnected</span>
    </header>
    <div class="main-content">
        <div id="map"></div>
        <div class="sidebar">
            <div class="sidebar-header">
                <h2>Exclusion Zones</h2>
            </div>
            <div class="zone-list" id="zone-list">
                <div class="empty-state" id="empty-state">
                    <p>No exclusion zones defined.</p>
                    <p>Use the polygon tool on the map to draw one.</p>
                </div>
            </div>
        </div>
    </div>
    <script>
        var ZONE_COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#8b5cf6', '#ec4899', '#6366f1'];

        // Initialize map
        var map = L.map('map').setView([{{CENTER_LAT}}, {{CENTER_LON}}], {{ZOOM}});
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);

        // Drawing layer
        var drawnItems = new L.FeatureGroup();
        map.addLayer(drawnItems);

        var drawControl = new L.Control.Draw({
            draw: {
                polygon: {
                    allowIntersection: false,
                    shapeOptions: { color: '#ef4444', fillOpacity: 0.2 }
                },
                polyline: false,
                rectangle: false,
                circle: false,
                circlemarker: false,
                marker: false
            },
            edit: {
                featureGroup: drawnItems,
                remove: false
            }
        });
        map.addControl(drawControl);

        // State
        var ws = null;
        var zones = {};  // id -> {layer, data}

        function getColor(id) {
            return ZONE_COLORS[id % ZONE_COLORS.length];
        }

        // WebSocket connection
        function connect() {
            ws = new WebSocket('ws://localhost:{{WS_PORT}}');

            ws.onopen = function() {
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').classList.remove('disconnected');
                ws.send(JSON.stringify({type: 'get_zones'}));
            };

            ws.onclose = function() {
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').classList.add('disconnected');
                setTimeout(connect, 2000);
            };

            ws.onerror = function() {};

            ws.onmessage = function(event) {
                var msg = JSON.parse(event.data);
                handleMessage(msg);
            };
        }

        function handleMessage(msg) {
            switch(msg.type) {
                case 'zones_list':
                    loadZones(msg.data);
                    break;
                case 'zone_saved':
                    addZoneToMap(msg.data);
                    updateSidebar();
                    break;
                case 'zone_updated':
                    updateZoneOnMap(msg.data);
                    updateSidebar();
                    break;
                case 'zone_deleted':
                    removeZoneFromMap(msg.data.id);
                    updateSidebar();
                    break;
                case 'error':
                    alert('Error: ' + msg.data.message);
                    break;
            }
        }

        function loadZones(zoneList) {
            // Clear existing
            Object.keys(zones).forEach(function(id) {
                drawnItems.removeLayer(zones[id].layer);
            });
            zones = {};

            zoneList.forEach(function(z) {
                addZoneToMap(z);
            });
            updateSidebar();

            // Fit bounds if zones exist
            if (zoneList.length > 0 && drawnItems.getLayers().length > 0) {
                map.fitBounds(drawnItems.getBounds(), {padding: [50, 50]});
            }
        }

        function addZoneToMap(z) {
            var color = getColor(z.id);
            var latlngs = z.polygon.map(function(p) { return [p[0], p[1]]; });
            var layer = L.polygon(latlngs, {
                color: color,
                fillColor: color,
                fillOpacity: 0.2,
                weight: 2
            });
            layer.zoneId = z.id;
            drawnItems.addLayer(layer);
            zones[z.id] = {layer: layer, data: z};
        }

        function updateZoneOnMap(z) {
            if (zones[z.id]) {
                drawnItems.removeLayer(zones[z.id].layer);
            }
            addZoneToMap(z);
        }

        function removeZoneFromMap(id) {
            if (zones[id]) {
                drawnItems.removeLayer(zones[id].layer);
                delete zones[id];
            }
        }

        function updateSidebar() {
            var list = document.getElementById('zone-list');
            var empty = document.getElementById('empty-state');
            var ids = Object.keys(zones);

            if (ids.length === 0) {
                list.innerHTML = '';
                list.appendChild(empty);
                empty.style.display = 'block';
                return;
            }

            list.innerHTML = '';
            ids.forEach(function(id) {
                var z = zones[id].data;
                var color = getColor(z.id);
                var vertices = z.polygon.length;

                var item = document.createElement('div');
                item.className = 'zone-item';
                item.innerHTML =
                    '<div class="zone-color" style="background:' + color + '"></div>' +
                    '<div class="zone-info">' +
                        '<div class="zone-name">' + escapeHtml(z.name) + '</div>' +
                        '<div class="zone-meta">' + vertices + ' vertices</div>' +
                    '</div>' +
                    '<div class="zone-actions">' +
                        '<button class="zone-btn rename" data-id="' + z.id + '" title="Rename">Rename</button>' +
                        '<button class="zone-btn delete" data-id="' + z.id + '" title="Delete">Delete</button>' +
                    '</div>';

                // Click to pan
                item.addEventListener('click', function(e) {
                    if (e.target.tagName === 'BUTTON') return;
                    var layer = zones[id].layer;
                    map.fitBounds(layer.getBounds(), {padding: [50, 50]});
                });

                list.appendChild(item);
            });

            // Bind button events
            list.querySelectorAll('.zone-btn.rename').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    var zId = parseInt(this.dataset.id);
                    var currentName = zones[zId].data.name;
                    var newName = prompt('Rename zone:', currentName);
                    if (newName && newName !== currentName) {
                        ws.send(JSON.stringify({
                            type: 'update_zone',
                            data: {id: zId, name: newName}
                        }));
                    }
                });
            });

            list.querySelectorAll('.zone-btn.delete').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    var zId = parseInt(this.dataset.id);
                    var name = zones[zId].data.name;
                    if (confirm('Delete zone "' + name + '"?')) {
                        ws.send(JSON.stringify({
                            type: 'delete_zone',
                            data: {id: zId}
                        }));
                    }
                });
            });
        }

        function escapeHtml(text) {
            var div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Handle new polygon drawn
        map.on(L.Draw.Event.CREATED, function(event) {
            var layer = event.layer;
            var latlngs = layer.getLatLngs()[0];
            var polygon = latlngs.map(function(ll) { return [ll.lat, ll.lng]; });

            var name = prompt('Name for this exclusion zone:', 'Unnamed Zone');
            if (name === null) return;  // Cancelled
            if (!name) name = 'Unnamed Zone';

            ws.send(JSON.stringify({
                type: 'save_zone',
                data: {name: name, polygon: polygon}
            }));
        });

        // Handle polygon edited
        map.on(L.Draw.Event.EDITED, function(event) {
            event.layers.eachLayer(function(layer) {
                if (layer.zoneId !== undefined) {
                    var latlngs = layer.getLatLngs()[0];
                    var polygon = latlngs.map(function(ll) { return [ll.lat, ll.lng]; });
                    ws.send(JSON.stringify({
                        type: 'update_zone',
                        data: {id: layer.zoneId, polygon: polygon}
                    }));
                }
            });
        });

        connect();
    </script>
</body>
</html>'''


class ZoneEditorServer:
    """HTTP and WebSocket server for the exclusion zone editor."""

    def __init__(self, history: HistoryDB, center: Optional[tuple[float, float]] = None,
                 http_port: int = 8090, ws_port: int = 8766):
        self.history = history
        self.center = center
        self.http_port = http_port
        self.ws_port = ws_port
        self.connected_clients: set = set()
        self._running = False
        self.ws_loop = None

    def start(self):
        """Start the server and block until Ctrl+C."""
        self._running = True

        # Start HTTP server
        http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        http_thread.start()

        # Start WebSocket server
        ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        ws_thread.start()

        time.sleep(0.5)

        url = f"http://localhost:{self.http_port}"
        print(f"Zone editor available at: {url}")
        webbrowser.open(url)

        print("Press Ctrl+C to stop the zone editor.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nZone editor stopped.")
            self._running = False

    def _get_map_center_and_zoom(self) -> tuple[float, float, int]:
        """Determine map center and zoom level."""
        if self.center:
            return self.center[0], self.center[1], 15

        # Try to center on existing zones
        zones = self.history.get_exclusion_zones()
        if zones:
            all_lats = []
            all_lons = []
            for z in zones:
                for point in z["polygon"]:
                    all_lats.append(point[0])
                    all_lons.append(point[1])
            center_lat = (min(all_lats) + max(all_lats)) / 2
            center_lon = (min(all_lons) + max(all_lons)) / 2
            return center_lat, center_lon, 14

        # Default: world view
        return 0, 0, 2

    def _run_http_server(self):
        """Run the HTTP server."""
        center_lat, center_lon, zoom = self._get_map_center_and_zoom()
        handler = partial(_ZoneEditorHTTPHandler, self.ws_port, center_lat, center_lon, zoom)
        with socketserver.TCPServer(("", self.http_port), handler) as httpd:
            httpd.allow_reuse_address = True
            while self._running:
                httpd.handle_request()

    def _run_ws_server(self):
        """Run the WebSocket server."""
        self.ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.ws_loop)

        async def handler(websocket):
            self.connected_clients.add(websocket)
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_ws_message(data, websocket)
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

    async def _handle_ws_message(self, msg: dict, websocket):
        """Handle a WebSocket message from the client."""
        msg_type = msg.get("type")

        if msg_type == "get_zones":
            zones = self.history.get_exclusion_zones()
            await websocket.send(json.dumps({"type": "zones_list", "data": zones}))

        elif msg_type == "save_zone":
            data = msg.get("data", {})
            name = data.get("name", "Unnamed Zone")
            polygon = data.get("polygon", [])
            if not polygon or len(polygon) < 3:
                await websocket.send(json.dumps({
                    "type": "error",
                    "data": {"message": "Polygon must have at least 3 vertices"}
                }))
                return
            zone_id = self.history.add_exclusion_zone(name, polygon)
            zone = {
                "id": zone_id,
                "name": name,
                "polygon": polygon,
            }
            await self._broadcast(json.dumps({"type": "zone_saved", "data": zone}))

        elif msg_type == "update_zone":
            data = msg.get("data", {})
            zone_id = data.get("id")
            if zone_id is None:
                return
            name = data.get("name")
            polygon = data.get("polygon")
            self.history.update_exclusion_zone(zone_id, name=name, polygon=polygon)
            # Re-fetch full zone to send back
            zones = self.history.get_exclusion_zones()
            updated = next((z for z in zones if z["id"] == zone_id), None)
            if updated:
                await self._broadcast(json.dumps({"type": "zone_updated", "data": updated}))

        elif msg_type == "delete_zone":
            data = msg.get("data", {})
            zone_id = data.get("id")
            if zone_id is None:
                return
            self.history.delete_exclusion_zone(zone_id)
            await self._broadcast(json.dumps({"type": "zone_deleted", "data": {"id": zone_id}}))

    async def _broadcast(self, message: str):
        """Send a message to all connected clients."""
        for client in list(self.connected_clients):
            try:
                await client.send(message)
            except Exception:
                self.connected_clients.discard(client)


class _ZoneEditorHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves the zone editor UI."""

    def __init__(self, ws_port: int, center_lat: float, center_lon: float,
                 zoom: int, *args, **kwargs):
        self.ws_port = ws_port
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = ZONE_EDITOR_HTML
            html = html.replace('{{WS_PORT}}', str(self.ws_port))
            html = html.replace('{{CENTER_LAT}}', str(self.center_lat))
            html = html.replace('{{CENTER_LON}}', str(self.center_lon))
            html = html.replace('{{ZOOM}}', str(self.zoom))
            self.wfile.write(html.encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress HTTP log messages
