#!/usr/bin/env python3
"""
DriveEasy Eye - Advanced AI Smart Dashcam v4.0

SETUP:
  export GEMINI_API_KEY="your_key"
  python3 webrtc_ai_sender.py

  Map UI: python3 web.py  (open http://PI_IP:5000 on phone/browser)

v4.0 Changes:
  * Logo generated programmatically in code — no external file needed
  * GPS writes location.json every 2 sec for live map in web.py
  * All previous features intact
"""

import asyncio
import cv2
import numpy as np
from picamera2 import Picamera2
import json
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc import RTCConfiguration, RTCIceServer
from av import VideoFrame
import time
from datetime import datetime
import os
import threading
import re
import collections
from http.server import BaseHTTPRequestHandler, HTTPServer
import google.generativeai as genai

# ── Constants ──────────────────────────────────────────────────────────────────
SIGNALING_SERVER = "wss://driveeasy-eye-signaling-904892438797.us-central1.run.app/"
DEVICE_ID        = "AICAM1"
EVENT_FOLDER     = "events"
HTTP_PORT        = 8080
LOCATION_FILE    = "/home/driveeasy/ai_dashcam/location.json"

FOCAL_LENGTH     = 700

KNOWN_HEIGHTS = {
    "person":    1.7,
    "car":       1.5,
    "truck":     3.5,
    "bus":       3.2,
    "motorbike": 1.2,
}

VEHICLE_LABELS = {"car", "truck", "bus", "motorbike"}

if not os.path.exists(EVENT_FOLDER):
    os.makedirs(EVENT_FOLDER)


# ==========================
# LOGO GENERATOR (no file needed)
# ==========================

def generate_logo(height=40):
    """
    Generates the DriveEasy 'D' logo programmatically using OpenCV.
    Blue rounded-D shape on black background.
    Returns (logo_bgr, logo_mask) — ready to overlay on header.
    """
    size  = height
    img   = np.zeros((size, size, 3), dtype=np.uint8)
    cx    = size // 2
    cy    = size // 2
    blue  = (200, 100, 0)   # BGR blue
    thick = max(2, size // 14)

    # Outer D arc (right half circle)
    cv2.ellipse(img, (cx - size//10, cy),
                (size//2 - thick, size//2 - thick),
                0, -75, 75, blue, thick)

    # Horizontal lines (left side of D)
    line_xs = size // 6
    line_xe = cx + size // 8
    gaps     = 5
    for i in range(gaps):
        y = int(thick + i * (size - 2 * thick) / (gaps - 1))
        cv2.line(img, (line_xs, y), (line_xe, y), blue, thick)
        # Rounded caps
        cv2.circle(img, (line_xs, y), thick // 2, blue, -1)
        cv2.circle(img, (line_xe, y), thick // 2, blue, -1)

    # Mask: any non-black pixel is logo
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    print(f"Logo generated: {size}x{size}px (programmatic)")
    return img, mask


def overlay_logo(frame, logo, mask, x_offset=5, y_offset=5):
    """Overlays logo onto frame. Non-black logo pixels replace frame pixels."""
    if logo is None:
        return
    lh, lw = logo.shape[:2]
    roi = frame[y_offset:y_offset + lh, x_offset:x_offset + lw]
    if roi.shape[:2] != (lh, lw):
        return
    logo_area = cv2.bitwise_and(logo, logo, mask=mask)
    bg_area   = cv2.bitwise_and(roi,  roi,  mask=cv2.bitwise_not(mask))
    frame[y_offset:y_offset + lh, x_offset:x_offset + lw] = cv2.add(logo_area, bg_area)


# ==========================
# VOICE ALERT
# ==========================

last_voice_time = 0

def speak(msg):
    global last_voice_time
    if time.time() - last_voice_time > 5:
        os.system(f'espeak "{msg}" &')
        last_voice_time = time.time()


# ==========================
# EVENT SNAPSHOT
# ==========================

def save_event(frame):
    filename = datetime.now().strftime("event_%Y%m%d_%H%M%S.jpg")
    cv2.imwrite(os.path.join(EVENT_FOLDER, filename), frame)


# ==========================
# CPU TEMPERATURE
# ==========================

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return f"{int(f.read().strip()) / 1000:.0f}C"
    except Exception:
        return "--"


# ==========================
# GPS READER + LOCATION WRITER
# ==========================

class GPSReader:
    """
    Reads from gpsd. Falls back silently if not available.
    Writes location.json every 2 seconds for the web map.
    Install: sudo apt install gpsd gpsd-clients python3-gps
    Connect: NEO-6M TX→Pin10, RX→Pin8, VCC→Pin1(3.3V), GND→Pin6
    """
    def __init__(self):
        self.speed_kmh = None
        self.lat       = None
        self.lon       = None
        self.enabled   = False
        self._lock     = threading.Lock()
        try:
            import gps as gpslib
            self.gpslib  = gpslib
            self.session = gpslib.gps(
                mode=gpslib.WATCH_ENABLE | gpslib.WATCH_NEWSTYLE
            )
            self.enabled = True
            threading.Thread(target=self._loop, daemon=True).start()
            print("GPS reader started — writing location.json")
        except Exception:
            print("GPS not available — map will show placeholder location")
            self._write_placeholder()

    def _loop(self):
        last_write = 0
        while True:
            try:
                report = self.session.next()
                if report["class"] == "TPV":
                    spd = getattr(report, "speed", None)
                    lat = getattr(report, "lat",   None)
                    lon = getattr(report, "lon",   None)
                    with self._lock:
                        if spd is not None:
                            self.speed_kmh = round(spd * 3.6, 1)
                        if lat is not None:
                            self.lat = lat
                        if lon is not None:
                            self.lon = lon
                    # Write location.json every 2 seconds
                    if time.time() - last_write > 2:
                        self._write_location()
                        last_write = time.time()
            except Exception:
                time.sleep(1)

    def _write_location(self):
        with self._lock:
            data = {
                "lat":   self.lat   if self.lat   else 0,
                "lon":   self.lon   if self.lon   else 0,
                "speed": self.speed_kmh if self.speed_kmh else 0,
                "valid": self.lat is not None and self.lon is not None,
                "time":  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        try:
            with open(LOCATION_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _write_placeholder(self):
        """Write Hyderabad coords as placeholder when GPS not connected."""
        try:
            with open(LOCATION_FILE, "w") as f:
                json.dump({
                    "lat": 17.3850, "lon": 78.4867,
                    "speed": 0, "valid": False,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f)
        except Exception:
            pass

    def get_speed(self):
        with self._lock:
            if not self.enabled or self.speed_kmh is None:
                return "--"
            return f"{self.speed_kmh}km/h"

    def get_coords(self):
        with self._lock:
            return self.lat, self.lon


# ==========================
# MANUAL RECORD CONTROLLER
# ==========================

class ManualRecorder:
    """
    HTTP toggle: http://PI_IP:8080/record
    http://PI_IP:8080/status
    """
    def __init__(self, fps=15):
        self.fps        = fps
        self.recording  = False
        self.start_time = None
        self._lock      = threading.Lock()
        self._frames    = []

    def start_http_server(self):
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/record":
                    msg = recorder.toggle()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(msg.encode())
                elif self.path == "/status":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"recording": recorder.recording}).encode()
                    )
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("0.0.0.0", HTTP_PORT), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Manual record: http://0.0.0.0:{HTTP_PORT}/record")

    def toggle(self):
        with self._lock:
            if not self.recording:
                self.recording  = True
                self.start_time = time.time()
                self._frames    = []
                return "Recording started"
            else:
                self.recording = False
                frames = list(self._frames)
                self._frames = []
                threading.Thread(
                    target=self._write, args=(frames,), daemon=True
                ).start()
                return "Recording stopped — saving..."

    def add_frame(self, frame):
        with self._lock:
            if self.recording:
                self._frames.append(frame.copy())

    def _write(self, frames):
        if not frames:
            return
        path = os.path.join(
            EVENT_FOLDER,
            datetime.now().strftime("manual_%Y%m%d_%H%M%S.avi")
        )
        h, w = frames[0].shape[:2]
        out  = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"XVID"), self.fps, (w, h)
        )
        for f in frames:
            out.write(f)
        out.release()
        print(f"Manual clip saved: {path}")

    def rec_duration(self):
        if not self.recording or self.start_time is None:
            return ""
        e = int(time.time() - self.start_time)
        return f"{e // 60:02d}:{e % 60:02d}"


# ==========================
# DISTANCE ESTIMATOR
# ==========================

class DistanceEstimator:
    def __init__(self, fl=FOCAL_LENGTH):
        self.fl = fl

    def estimate(self, label, pixel_height):
        if pixel_height <= 0:
            return None
        rh = KNOWN_HEIGHTS.get(label)
        if rh is None:
            return None
        return round((rh * self.fl) / pixel_height, 1)

    def nearest_vehicle(self, detections):
        best_d, best_l = None, None
        for (x, y, w, h, label) in detections:
            if label not in VEHICLE_LABELS:
                continue
            d = self.estimate(label, h)
            if d is not None and (best_d is None or d < best_d):
                best_d, best_l = d, label
        return best_d, best_l

    def nearest_any(self, detections):
        best_d, best_l = None, None
        for (x, y, w, h, label) in detections:
            d = self.estimate(label, h)
            if d is not None and (best_d is None or d < best_d):
                best_d, best_l = d, label
        return best_d, best_l


# ==========================
# NIGHT MODE ENHANCER
# ==========================

class NightModeEnhancer:
    def __init__(self, threshold=60):
        self.threshold  = threshold
        self.clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.night_mode = False

    def process(self, frame):
        self.night_mode = (
            np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < self.threshold
        )
        if self.night_mode:
            lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            lab     = cv2.merge((self.clahe.apply(l), a, b))
            frame   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return frame, self.night_mode


# ==========================
# FATIGUE MONITOR
# ==========================

class FatigueMonitor:
    def __init__(self, warn_after_minutes=120, remind_every_minutes=30):
        self.start      = time.time()
        self.warn_after = warn_after_minutes * 60
        self.remind     = remind_every_minutes * 60
        self.last_warn  = 0

    def check(self):
        elapsed = time.time() - self.start
        if elapsed >= self.warn_after:
            if time.time() - self.last_warn >= self.remind:
                self.last_warn = time.time()
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                return f"Fatigue Alert {h}h{m:02d}m Driving"
        return None

    def uptime(self):
        e = int(time.time() - self.start)
        return f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}"


# ==========================
# RAPID APPROACH DETECTOR
# ==========================

class RapidApproachDetector:
    def __init__(self, growth=0.20):
        self.prev   = {}
        self.growth = growth

    def check(self, detections):
        curr = {}
        for (x, y, w, h, label) in detections:
            if label in VEHICLE_LABELS:
                if label not in curr or h > curr[label]:
                    curr[label] = h
        warns = []
        for label, ch in curr.items():
            if label in self.prev and self.prev[label] > 0:
                if ch > self.prev[label] * (1 + self.growth):
                    warns.append("Vehicle Approaching Fast")
                    break
        self.prev = curr
        return warns


# ==========================
# AUTO EVENT VIDEO BUFFER
# ==========================

class EventVideoBuffer:
    def __init__(self, fps=15, seconds=10):
        self.fps     = fps
        self.buf     = collections.deque(maxlen=fps * seconds)
        self._saving = False

    def add(self, frame):
        self.buf.append(frame.copy())

    def save(self, tag="event"):
        if self._saving:
            return
        frames       = list(self.buf)
        self._saving = True
        threading.Thread(
            target=self._write, args=(frames, tag), daemon=True
        ).start()

    def _write(self, frames, tag):
        if not frames:
            self._saving = False
            return
        path = os.path.join(
            EVENT_FOLDER,
            datetime.now().strftime(f"{tag}_%Y%m%d_%H%M%S.avi")
        )
        h, w = frames[0].shape[:2]
        out  = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"XVID"), self.fps, (w, h)
        )
        for f in frames:
            out.write(f)
        out.release()
        print(f"Auto event saved: {path}")
        self._saving = False


# ==========================
# GEMINI AI (FREE)
# ==========================

class GeminiVisionAnalyzer:
    def __init__(self):
        key = os.environ.get("GEMINI_API_KEY", "AIzaSyAur7MU_a4P7FJxh2vfCj5Y6qUObPvdOhQ")
        if not key:
            print("GEMINI_API_KEY not set — Gemini AI disabled.")
            self.enabled = False
            return
        genai.configure(api_key=key)
        self.model   = genai.GenerativeModel("gemini-1.5-flash")
        self.warns   = []
        self._lock   = threading.Lock()
        self._run    = False
        self._frame  = None
        self.enabled = True
        print("Gemini Vision AI ready (FREE tier)")

    def submit(self, frame):
        if not self.enabled:
            return
        with self._lock:
            self._frame = frame.copy()

    def start(self):
        if not self.enabled:
            return
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._run = False

    def _loop(self):
        while self._run:
            frame = None
            with self._lock:
                if self._frame is not None:
                    frame, self._frame = self._frame, None
            if frame is not None:
                try:
                    self._analyze(frame)
                except Exception as e:
                    print(f"Gemini error: {e}")
            time.sleep(6)

    def _analyze(self, frame):
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        prompt = (
            "You are an AI dashcam safety system. "
            "Analyze this dashcam image and return ONLY a JSON array of short critical driving "
            "warnings (max 3 items). Focus on road hazards, obstacles, unsafe behavior, weather, "
            "pedestrians in road, unusual events. Return [] if nothing notable. "
            "Example: [\"Road debris ahead\", \"Wet road surface\"] "
            "Return ONLY the JSON array. No explanation, no markdown."
        )
        resp  = self.model.generate_content([
            {"mime_type": "image/jpeg", "data": buf.tobytes()}, prompt
        ])
        match = re.search(r'\[.*?\]', resp.text.strip(), re.DOTALL)
        with self._lock:
            self.warns = (
                [str(w) for w in json.loads(match.group())[:3]]
                if match else []
            )

    def get(self):
        if not self.enabled:
            return []
        with self._lock:
            return list(self.warns)


# ==========================
# FPS COUNTER
# ==========================

class FPSCounter:
    def __init__(self, window=30):
        self.times = collections.deque(maxlen=window)
        self.last  = time.time()

    def tick(self):
        now = time.time()
        self.times.append(now - self.last)
        self.last = now

    def fps(self):
        if len(self.times) < 2:
            return 0.0
        return round(1.0 / (sum(self.times) / len(self.times)), 1)


# ==========================
# LANE DETECTION (FIXED)
# ==========================

def detect_lane(frame, vehicles_present):
    if not vehicles_present:
        return None

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

    fh, fw = frame.shape[:2]
    mask   = np.zeros_like(edges)
    poly   = np.array([[
        (0,  fh),
        (fw, fh),
        (fw, int(fh * 0.65)),
        (0,  int(fh * 0.65))
    ]], np.int32)
    cv2.fillPoly(mask, poly, 255)

    raw_lines = cv2.HoughLinesP(
        cv2.bitwise_and(edges, mask),
        2, np.pi / 180, 100,
        minLineLength=80, maxLineGap=40
    )

    if raw_lines is None:
        return None

    filtered = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if not (70 <= angle <= 110):
            filtered.append(line)

    return filtered if filtered else None


# ==========================
# COLLISION WARNING
# ==========================

def collision_warning_vehicle(box_height, frame_height, label):
    if label not in VEHICLE_LABELS:
        return "SAFE"
    ratio = box_height / frame_height
    if ratio > 0.40:
        return "HIGH"
    if ratio > 0.25:
        return "MEDIUM"
    return "SAFE"


# ==========================
# AI DECISION ENGINE
# ==========================

def decision(persons, vehicles, traffic, signal, nearest_veh_dist):
    warns = []
    if persons > 0:
        warns.append("Pedestrian Ahead")
    if traffic == "HIGH":
        warns.append("Heavy Traffic")
    if signal == "RED":
        warns.append("Stop Vehicle")
    if vehicles > 5:
        warns.append("Maintain Safe Distance")
    if nearest_veh_dist is not None:
        if nearest_veh_dist < 5.0:
            warns.append(f"TAILGATING! Only {nearest_veh_dist}m")
        elif nearest_veh_dist < 10.0:
            warns.append(f"Too Close: {nearest_veh_dist}m")
    return warns


# ==========================
# YOLO AI ENGINE
# ==========================

class DriveEasyAI:

    def __init__(self):
        print("Loading YOLO model...")
        self.net = cv2.dnn.readNet(
            "models/yolov4-tiny.weights",
            "models/yolov4-tiny.cfg"
        )
        with open("coco.names", "r") as f:
            self.classes = [l.strip() for l in f.readlines()]
        self.layer_names   = self.net.getLayerNames()
        self.output_layers = [
            self.layer_names[i - 1]
            for i in self.net.getUnconnectedOutLayers()
        ]
        self.signal_color = None
        print("AI Ready")

    def detect_signal_color(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0,   100, 100), (10,  255, 255)),
            cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        )
        yellow = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        green  = cv2.inRange(hsv, (40,  50,  50), (90, 255, 255))
        r = cv2.countNonZero(red)
        y = cv2.countNonZero(yellow)
        g = cv2.countNonZero(green)
        if max(r, y, g) < 50:
            return None
        if r > y and r > g:
            return "RED"
        if y > r and y > g:
            return "YELLOW"
        return "GREEN"

    def detect_objects(self, frame):
        fh, fw = frame.shape[:2]
        blob   = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (320, 320), swapRB=True
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confs, cids = [], [], []
        for out in outputs:
            for det in out:
                sc  = det[5:]
                cid = np.argmax(sc)
                cf  = sc[cid]
                if cf > 0.35:
                    cx = int(det[0] * fw)
                    cy = int(det[1] * fh)
                    w  = int(det[2] * fw)
                    h  = int(det[3] * fh)
                    boxes.append([cx - w // 2, cy - h // 2, w, h])
                    confs.append(float(cf))
                    cids.append(cid)

        idx        = cv2.dnn.NMSBoxes(boxes, confs, 0.35, 0.4)
        persons    = 0
        vehicles   = 0
        detections = []

        if len(idx) > 0:
            for i in idx.flatten():
                x, y, w, h = boxes[i]
                label       = self.classes[cids[i]]
                detections.append((x, y, w, h, label))
                if label == "person":
                    persons += 1
                if label in VEHICLE_LABELS:
                    vehicles += 1
                if label == "traffic light":
                    c = self.detect_signal_color(frame[y:y+h, x:x+w])
                    if c:
                        self.signal_color = c

        traffic = "LOW"
        if vehicles >= 6:
            traffic = "HIGH"
        elif vehicles >= 3:
            traffic = "MEDIUM"

        return detections, persons, vehicles, traffic, self.signal_color


# ==========================
# VIDEO STREAM
# ==========================

class AIVideoStreamTrack(VideoStreamTrack):

    def __init__(self, camera, ai, gemini_ai, recorder, gps):
        super().__init__()
        self.camera   = camera
        self.ai       = ai
        self.gemini   = gemini_ai
        self.recorder = recorder
        self.gps      = gps

        self.dist_est   = DistanceEstimator()
        self.night      = NightModeEnhancer()
        self.fatigue    = FatigueMonitor()
        self.approach   = RapidApproachDetector()
        self.evt_buf    = EventVideoBuffer()
        self.fps_ctr    = FPSCounter()

        # Generate logo once at startup — no file needed
        self.logo, self.logo_mask = generate_logo(height=40)

        self.frame_count     = 0
        self.cached          = ([], 0, 0, "LOW", None)
        self._last_collision = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = cv2.cvtColor(self.camera.capture_array(), cv2.COLOR_RGB2BGR)

        frame, is_night = self.night.process(frame)
        fh, fw = frame.shape[:2]

        if self.frame_count % 5 == 0:
            self.cached = self.ai.detect_objects(frame)

        detections, persons, vehicles, traffic, signal_color = self.cached

        nearest_veh_dist, _ = self.dist_est.nearest_vehicle(detections)
        nearest_any_dist, _ = self.dist_est.nearest_any(detections)

        warns = decision(persons, vehicles, traffic, signal_color, nearest_veh_dist)

        if self.frame_count % 5 == 0:
            warns.extend(self.approach.check(detections))

        fw_warn = self.fatigue.check()
        if fw_warn:
            warns.append(fw_warn)

        if is_night:
            warns.append("Low Light: Night Mode Active")

        for gw in self.gemini.get():
            if gw not in warns:
                warns.append(gw)

        if self.frame_count % 90 == 0:
            self.gemini.submit(frame)

        # Lane detection (vehicle-gated + angle-filtered)
        lanes = detect_lane(frame, vehicles > 0)
        if lanes is not None:
            for ln in lanes:
                x1, y1, x2, y2 = ln[0]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Bounding boxes + distance label
        for (x, y, w, h, label) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            d    = self.dist_est.estimate(label, h)
            disp = f"{label} {d}m" if d is not None else label
            cv2.putText(frame, disp, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            risk = collision_warning_vehicle(h, fh, label)
            if risk == "HIGH" and time.time() - self._last_collision > 5:
                warns.append("Collision Risk")
                save_event(frame)
                self.evt_buf.save("collision")
                self._last_collision = time.time()

        if self.frame_count % 2 == 0:
            self.evt_buf.add(frame)
        self.recorder.add_frame(frame)

        # Warning overlay
        for i, w in enumerate(warns):
            cv2.putText(frame, w,
                        (30, 80 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
            speak(w)

        # System info
        self.fps_ctr.tick()
        fps_val  = self.fps_ctr.fps()
        cpu_temp = get_cpu_temp()
        uptime   = self.fatigue.uptime()
        spd      = self.gps.get_speed()
        lat, lon = self.gps.get_coords()
        now      = datetime.now()

        # ── HEADER ────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (fw, 50), (0, 0, 0), -1)

        # Logo — programmatically generated, always works
        overlay_logo(frame, self.logo, self.logo_mask, x_offset=5, y_offset=5)

        # Title shifted right of logo
        cv2.putText(frame, "DriveEasy Eye AI Dashcam",
                    (52, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Night badge
        if is_night:
            cv2.putText(frame, "[NIGHT]",
                        (fw - 245, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

        # REC indicator
        if self.recorder.recording:
            dur = self.recorder.rec_duration()
            cv2.circle(frame, (fw - 95, 25), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {dur}",
                        (fw - 80, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # ── FOOTER (2-line) ───────────────────────────────────────────
        cv2.rectangle(frame, (0, fh - 50), (fw, fh), (0, 0, 0), -1)

        # Line 1 left — timestamp
        cv2.putText(frame,
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    (20, fh - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Line 1 right — system info
        sys_info = f"FPS:{fps_val}  CPU:{cpu_temp}  Up:{uptime}  Spd:{spd}"
        cv2.putText(frame, sys_info,
                    (fw - 520, fh - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)

        # Line 1 right-most — GPS coords (if available)
        if lat is not None and lon is not None:
            gps_str = f"{lat:.4f},{lon:.4f}"
            cv2.putText(frame, gps_str,
                        (fw - 200, fh - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1)

        # Line 2 centre — detection telemetry
        dist_str  = f"{nearest_any_dist}m" if nearest_any_dist is not None else "--"
        telemetry = (
            f"Persons:{persons}  Vehicles:{vehicles}  "
            f"Traffic:{traffic}  Dist:{dist_str}"
        )
        if signal_color:
            telemetry += f"  Signal:{signal_color}"
        cv2.putText(frame, telemetry,
                    (fw // 2 - 330, fh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)

        self.frame_count += 1

        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts, vf.time_base = pts, time_base
        return vf


# ==========================
# WEBRTC
# ==========================

async def run_webrtc_sender():

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(main={"size": (1280, 720)})
    )
    picam2.set_controls({"FrameRate": 30})
    picam2.start()
    time.sleep(2)

    ai       = DriveEasyAI()
    gemini   = GeminiVisionAnalyzer()
    gemini.start()
    gps      = GPSReader()
    recorder = ManualRecorder(fps=15)
    recorder.start_http_server()

    async with websockets.connect(SIGNALING_SERVER) as ws:

        await ws.send(json.dumps({
            "type": "register-pi", "deviceId": DEVICE_ID
        }))

        async for message in ws:
            data = json.loads(message)

            if data["type"] == "request-offer":
                pc = RTCPeerConnection(configuration=RTCConfiguration(
                    iceServers=[RTCIceServer(
                        urls=["stun:stun.l.google.com:19302"]
                    )]
                ))
                pc.addTrack(
                    AIVideoStreamTrack(picam2, ai, gemini, recorder, gps)
                )

                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)

                await ws.send(json.dumps({
                    "type":     "offer",
                    "deviceId": DEVICE_ID,
                    "sdp": {
                        "type": pc.localDescription.type,
                        "sdp":  pc.localDescription.sdp
                    }
                }))

            elif data["type"] == "answer":
                await pc.setRemoteDescription(RTCSessionDescription(
                    sdp=data["sdp"]["sdp"],
                    type=data["sdp"]["type"]
                ))


if __name__ == "__main__":
    asyncio.run(run_webrtc_sender())
