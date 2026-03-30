#!/usr/bin/env python3
"""
DriveEasy Eye - Web Dashboard
Runs on port 5000. Open http://PI_IP:5000 on any phone/browser.

Shows:
  - Live dashcam video (WebRTC embed)
  - Live location on OpenStreetMap (free, no API key)
  - Live telemetry: speed, GPS coords, session stats
  - Record button (triggers webrtc_ai_sender.py recording)
  - Trip trail on map

Run alongside main dashcam:
  python3 web.py

Install Flask if needed:
  pip install flask --break-system-packages
"""

from flask import Flask, jsonify, send_file, Response
import json
import os
import io
from datetime import datetime

app  = Flask(__name__)

LOCATION_FILE    = "/home/driveeasy/ai_dashcam/location.json"
SIGNALING_SERVER = "wss://driveeasy-eye-signaling-904892438797.us-central1.run.app/"
DEVICE_ID        = "AICAM1"
RECORD_PORT      = 8080


# ==========================
# API ENDPOINTS
# ==========================

@app.route("/api/location")
def api_location():
    try:
        with open(LOCATION_FILE, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception:
        return jsonify({
            "lat": 17.3850, "lon": 78.4867,
            "speed": 0, "valid": False,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })


@app.route("/api/record")
def api_record():
    """Proxy to dashcam recorder."""
    import urllib.request
    try:
        res = urllib.request.urlopen(
            f"http://localhost:{RECORD_PORT}/record", timeout=2
        )
        return jsonify({"ok": True, "msg": res.read().decode()})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/status")
def api_status():
    try:
        import urllib.request
        res  = urllib.request.urlopen(
            f"http://localhost:{RECORD_PORT}/status", timeout=2
        )
        data = json.loads(res.read().decode())
        return jsonify(data)
    except Exception:
        return jsonify({"recording": False})


# ==========================
# MAIN DASHBOARD PAGE
# ==========================

@app.route("/")
def index():
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DriveEasy Eye — Live Dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {{
    --blue:   #0A84FF;
    --cyan:   #00D4FF;
    --green:  #30D158;
    --red:    #FF453A;
    --amber:  #FFD60A;
    --bg:     #0A0A0F;
    --panel:  #13131A;
    --border: #1E1E2E;
    --text:   #E8E8F0;
    --muted:  #6B6B80;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Courier New', monospace;
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}

  /* ── TOP BAR ── */
  .topbar {{
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    height: 56px;
  }}

  .brand {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}

  /* Programmatic D logo in CSS */
  .logo-d {{
    width: 36px;
    height: 36px;
    position: relative;
    flex-shrink: 0;
  }}
  .logo-d svg {{ width: 100%; height: 100%; }}

  .brand-text {{
    font-size: 16px;
    font-weight: bold;
    color: var(--cyan);
    letter-spacing: 1px;
  }}
  .brand-sub {{
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
  }}

  .topbar-right {{
    display: flex;
    align-items: center;
    gap: 16px;
  }}

  .live-badge {{
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(48, 209, 88, 0.15);
    border: 1px solid var(--green);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: var(--green);
    font-weight: bold;
    letter-spacing: 1px;
  }}
  .live-dot {{
    width: 7px; height: 7px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse 1.4s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%       {{ opacity: 0.5; transform: scale(0.8); }}
  }}

  .time-display {{
    font-size: 13px;
    color: var(--cyan);
    letter-spacing: 1px;
  }}

  /* ── MAIN LAYOUT ── */
  .main {{
    display: grid;
    grid-template-columns: 1fr 380px;
    grid-template-rows: 1fr 120px;
    gap: 8px;
    padding: 8px;
    flex: 1;
    overflow: hidden;
    min-height: 0;
  }}

  /* ── VIDEO PANEL ── */
  .video-panel {{
    grid-row: 1 / 2;
    grid-column: 1 / 2;
    background: #000;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
  }}

  .video-panel iframe {{
    width: 100%;
    height: 100%;
    border: none;
  }}

  .video-label {{
    position: absolute;
    top: 10px;
    right: 12px;
    background: rgba(0,0,0,0.7);
    color: var(--cyan);
    font-size: 10px;
    letter-spacing: 2px;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid var(--border);
  }}

  /* ── MAP PANEL ── */
  .map-panel {{
    grid-row: 1 / 2;
    grid-column: 2 / 3;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    position: relative;
    display: flex;
    flex-direction: column;
  }}

  .map-header {{
    background: var(--panel);
    padding: 8px 14px;
    font-size: 11px;
    color: var(--cyan);
    letter-spacing: 2px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }}

  .gps-indicator {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 10px;
  }}
  .gps-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--muted);
    transition: background 0.4s;
  }}
  .gps-dot.active {{ background: var(--green); }}

  #map {{
    flex: 1;
    min-height: 0;
  }}

  .map-coords {{
    background: var(--panel);
    padding: 6px 14px;
    font-size: 10px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }}

  /* ── TELEMETRY BAR ── */
  .telemetry-bar {{
    grid-row: 2 / 3;
    grid-column: 1 / 3;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 0;
    overflow: hidden;
  }}

  .tele-item {{
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 8px 6px;
    border-right: 1px solid var(--border);
    min-width: 0;
  }}
  .tele-item:last-child {{ border-right: none; }}

  .tele-label {{
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }}

  .tele-value {{
    font-size: 20px;
    font-weight: bold;
    color: var(--text);
    letter-spacing: 1px;
    line-height: 1;
  }}

  .tele-value.speed  {{ color: var(--cyan);  }}
  .tele-value.green  {{ color: var(--green); }}
  .tele-value.amber  {{ color: var(--amber); }}
  .tele-value.red    {{ color: var(--red);   }}
  .tele-value.blue   {{ color: var(--blue);  }}

  .tele-unit {{
    font-size: 9px;
    color: var(--muted);
    margin-top: 2px;
  }}

  /* Record button */
  .rec-btn {{
    background: rgba(255,69,58,0.15);
    border: 1px solid var(--red);
    color: var(--red);
    border-radius: 8px;
    padding: 8px 18px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 2px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.2s;
    white-space: nowrap;
  }}
  .rec-btn:hover {{
    background: rgba(255,69,58,0.3);
  }}
  .rec-btn.recording {{
    background: rgba(255,69,58,0.4);
    animation: rec-flash 1s ease-in-out infinite;
  }}
  @keyframes rec-flash {{
    0%, 100% {{ border-color: var(--red); }}
    50%       {{ border-color: transparent; }}
  }}
  .rec-circle {{
    width: 10px; height: 10px;
    background: var(--red);
    border-radius: 50%;
  }}

  /* Map custom styles */
  .leaflet-container {{
    background: #1a1a2e !important;
    font-family: 'Courier New', monospace;
  }}
  .speed-popup {{
    background: var(--panel);
    color: var(--cyan);
    border: 1px solid var(--blue);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    font-family: monospace;
  }}
</style>
</head>
<body>

<!-- ── TOP BAR ── -->
<div class="topbar">
  <div class="brand">
    <!-- DriveEasy D logo in SVG — always renders, no file needed -->
    <div class="logo-d">
      <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path d="M15 10 Q75 10 85 50 Q75 90 15 90 Z" fill="none" stroke="#0064C8" stroke-width="9" stroke-linejoin="round"/>
        <line x1="14" y1="25" x2="65" y2="25" stroke="#0064C8" stroke-width="9" stroke-linecap="round"/>
        <line x1="14" y1="40" x2="73" y2="40" stroke="#0064C8" stroke-width="9" stroke-linecap="round"/>
        <line x1="14" y1="55" x2="76" y2="55" stroke="#0064C8" stroke-width="9" stroke-linecap="round"/>
        <line x1="14" y1="70" x2="72" y2="70" stroke="#0064C8" stroke-width="9" stroke-linecap="round"/>
        <line x1="14" y1="85" x2="62" y2="85" stroke="#0064C8" stroke-width="9" stroke-linecap="round"/>
      </svg>
    </div>
    <div>
      <div class="brand-text">DriveEasy Eye</div>
      <div class="brand-sub">AI DASHCAM · {DEVICE_ID}</div>
    </div>
  </div>

  <div class="topbar-right">
    <div class="live-badge">
      <div class="live-dot"></div>
      LIVE
    </div>
    <div class="time-display" id="clock">--:--:--</div>
  </div>
</div>

<!-- ── MAIN CONTENT ── -->
<div class="main">

  <!-- Video -->
  <div class="video-panel">
    <iframe
      src="https://driveeasy-eye-web-904892438797.us-central1.run.app"
      allow="camera; microphone; autoplay"
      allowfullscreen>
    </iframe>
    <div class="video-label">AICAM1 · 1280×720</div>
  </div>

  <!-- Map -->
  <div class="map-panel">
    <div class="map-header">
      <span>LIVE LOCATION</span>
      <div class="gps-indicator">
        <div class="gps-dot" id="gps-dot"></div>
        <span id="gps-status">LOCATING...</span>
      </div>
    </div>
    <div id="map"></div>
    <div class="map-coords">
      <span id="coord-lat">Lat: --</span>
      <span id="coord-lon">Lon: --</span>
      <span id="coord-time">--</span>
    </div>
  </div>

  <!-- Telemetry bar -->
  <div class="telemetry-bar">

    <div class="tele-item">
      <div class="tele-label">SPEED</div>
      <div class="tele-value speed" id="t-speed">--</div>
      <div class="tele-unit">km/h</div>
    </div>

    <div class="tele-item">
      <div class="tele-label">LATITUDE</div>
      <div class="tele-value blue" id="t-lat" style="font-size:14px">--</div>
      <div class="tele-unit">degrees</div>
    </div>

    <div class="tele-item">
      <div class="tele-label">LONGITUDE</div>
      <div class="tele-value blue" id="t-lon" style="font-size:14px">--</div>
      <div class="tele-unit">degrees</div>
    </div>

    <div class="tele-item">
      <div class="tele-label">GPS FIX</div>
      <div class="tele-value green" id="t-fix">--</div>
      <div class="tele-unit">status</div>
    </div>

    <div class="tele-item">
      <div class="tele-label">LAST UPDATE</div>
      <div class="tele-value" id="t-time" style="font-size:12px">--</div>
      <div class="tele-unit">timestamp</div>
    </div>

    <div class="tele-item">
      <div class="tele-label">TRAIL PTS</div>
      <div class="tele-value amber" id="t-trail">0</div>
      <div class="tele-unit">points</div>
    </div>

    <div class="tele-item" style="flex: 0 0 auto; padding: 0 20px;">
      <button class="rec-btn" id="rec-btn" onclick="toggleRecord()">
        <div class="rec-circle"></div>
        <span id="rec-label">RECORD</span>
      </button>
    </div>

  </div>
</div>

<script>
// ── CLOCK ──
function updateClock() {{
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('en-IN', {{hour12: false}});
}}
updateClock();
setInterval(updateClock, 1000);

// ── MAP SETUP ──
const map = L.map('map', {{
  zoomControl: true,
  attributionControl: false
}}).setView([17.3850, 78.4867], 15);

// Dark OpenStreetMap tile layer — free, no API key
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  subdomains: 'abcd',
  maxZoom: 19
}}).addTo(map);

// Custom marker icon
const markerIcon = L.divIcon({{
  className: '',
  html: `<div style="
    width:18px; height:18px;
    background:#00D4FF;
    border:3px solid #fff;
    border-radius:50%;
    box-shadow:0 0 12px #00D4FF;
  "></div>`,
  iconSize:   [18, 18],
  iconAnchor: [9, 9]
}});

const marker = L.marker([17.3850, 78.4867], {{icon: markerIcon}}).addTo(map);
marker.bindPopup('<div class="speed-popup">DriveEasy Eye · AICAM1</div>');

// Trail polyline
const trail = L.polyline([], {{
  color:   '#00D4FF',
  weight:  3,
  opacity: 0.8
}}).addTo(map);

let trailCount  = 0;
let isRecording = false;

// ── LOCATION POLLING ──
async function pollLocation() {{
  try {{
    const r    = await fetch('/api/location');
    const data = await r.json();

    const lat = parseFloat(data.lat);
    const lon = parseFloat(data.lon);
    const spd = parseFloat(data.speed) || 0;
    const valid = data.valid;

    // Update marker + trail
    const ll = [lat, lon];
    marker.setLatLng(ll);
    trail.addLatLng(ll);
    trailCount++;
    map.panTo(ll);

    // GPS indicator
    const dot = document.getElementById('gps-dot');
    const sta = document.getElementById('gps-status');
    if (valid) {{
      dot.classList.add('active');
      sta.textContent = 'GPS FIX';
    }} else {{
      dot.classList.remove('active');
      sta.textContent = 'NO FIX';
    }}

    // Telemetry values
    document.getElementById('t-speed').textContent = spd.toFixed(1);
    document.getElementById('t-lat').textContent   = lat.toFixed(5);
    document.getElementById('t-lon').textContent   = lon.toFixed(5);
    document.getElementById('t-fix').textContent   = valid ? 'LOCKED' : 'SEARCH';
    document.getElementById('t-fix').className     = 'tele-value ' + (valid ? 'green' : 'amber');
    document.getElementById('t-time').textContent  = data.time || '--';
    document.getElementById('t-trail').textContent = trailCount;

    // Coords bar
    document.getElementById('coord-lat').textContent  = 'Lat: ' + lat.toFixed(6);
    document.getElementById('coord-lon').textContent  = 'Lon: ' + lon.toFixed(6);
    document.getElementById('coord-time').textContent = data.time || '--';

    // Speed color
    const sv = document.getElementById('t-speed');
    sv.className = 'tele-value speed';
    if (spd > 80) sv.className = 'tele-value red';
    else if (spd > 50) sv.className = 'tele-value amber';

  }} catch (e) {{
    console.log('Location poll error:', e);
  }}
}}

pollLocation();
setInterval(pollLocation, 2000);

// ── RECORD TOGGLE ──
async function toggleRecord() {{
  const btn   = document.getElementById('rec-btn');
  const label = document.getElementById('rec-label');
  try {{
    const r    = await fetch('/api/record');
    const data = await r.json();
    isRecording = !isRecording;
    if (isRecording) {{
      btn.classList.add('recording');
      label.textContent = 'STOP REC';
    }} else {{
      btn.classList.remove('recording');
      label.textContent = 'RECORD';
    }}
  }} catch (e) {{
    console.log('Record error:', e);
  }}
}}

// ── CLEAR TRAIL ──
function clearTrail() {{
  trail.setLatLngs([]);
  trailCount = 0;
  document.getElementById('t-trail').textContent = '0';
}}

// Right-click map to clear trail
map.on('contextmenu', clearTrail);
</script>
</body>
</html>"""
    return html


if __name__ == "__main__":
    print("=" * 50)
    print("DriveEasy Eye Web Dashboard")
    print("Open on phone: http://PI_IP_ADDRESS:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
