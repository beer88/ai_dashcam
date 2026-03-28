#!/usr/bin/env python3
"""
DriveEasy Eye - Advanced AI Smart Dashcam

Features
* Person / Vehicle detection
* Traffic signal detection
* Traffic density estimation
* Lane detection
* Forward collision warning
* Smart AI decision engine (Google Gemini - FREE)
* Voice alerts
* Event recording
* Header + footer UI
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
import base64
import threading
import re
import google.generativeai as genai

SIGNALING_SERVER = "wss://driveeasy-eye-signaling-904892438797.us-central1.run.app/"
DEVICE_ID = "AICAM1"
EVENT_FOLDER = "events"

if not os.path.exists(EVENT_FOLDER):
    os.makedirs(EVENT_FOLDER)

# ==========================
# GEMINI AI ANALYZER (FREE)
# ==========================

class GeminiVisionAnalyzer:
    """
    Runs Google Gemini Vision analysis on dashcam frames
    in a background thread every N frames.
    Produces extra smart warnings without blocking video.
    Free tier: 1500 requests/day, 15 RPM.
    """

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("WARNING: GEMINI_API_KEY not set. Gemini AI disabled.")
            self.enabled = False
            return
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.extra_warnings = []
        self._lock = threading.Lock()
        self._running = False
        self._frame_to_analyze = None
        self.enabled = True
        print("Gemini Vision AI ready (FREE tier)")

    def submit_frame(self, frame):
        """Called from video loop — just stores frame, never blocks."""
        if not self.enabled:
            return
        with self._lock:
            self._frame_to_analyze = frame.copy()

    def start(self):
        if not self.enabled:
            return
        self._running = True
        t = threading.Thread(target=self._analysis_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _analysis_loop(self):
        while self._running:
            frame = None
            with self._lock:
                if self._frame_to_analyze is not None:
                    frame = self._frame_to_analyze
                    self._frame_to_analyze = None

            if frame is not None:
                try:
                    self._run_gemini(frame)
                except Exception as e:
                    print(f"Gemini error: {e}")

            # Sleep 6 seconds between calls to stay well under 15 RPM free limit
            time.sleep(6)

    def _run_gemini(self, frame):
        # Encode frame as JPEG bytes
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_bytes = buffer.tobytes()

        # Build prompt
        prompt = (
            "You are an AI dashcam safety system. "
            "Analyze this dashcam image and return ONLY a JSON array of short critical driving warnings (max 3 items). "
            "Focus on: road hazards, obstacles, unsafe behavior, weather conditions, pedestrians, unusual events. "
            "If nothing notable, return an empty array []. "
            "Example output: [\"Road debris ahead\", \"Wet road surface\"] "
            "Return ONLY the JSON array. No explanation, no markdown, no extra text."
        )

        # Send to Gemini with inline image bytes
        response = self.model.generate_content([
            {"mime_type": "image/jpeg", "data": img_bytes},
            prompt
        ])

        raw = response.text.strip()

        # Safe JSON parse — extract array even if model adds extra text
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            warnings = json.loads(match.group())
            with self._lock:
                self.extra_warnings = [str(w) for w in warnings[:3]]
        else:
            with self._lock:
                self.extra_warnings = []

    def get_warnings(self):
        if not self.enabled:
            return []
        with self._lock:
            return list(self.extra_warnings)


# ==========================
# VOICE ALERT
# ==========================

last_voice_time = 0


def speak(msg):
    global last_voice_time
    if time.time() - last_voice_time > 5:
        os.system(f'espeak "{msg}"')
        last_voice_time = time.time()


# ==========================
# EVENT RECORDING
# ==========================

def save_event(frame):
    filename = datetime.now().strftime("event_%Y%m%d_%H%M%S.jpg")
    path = os.path.join(EVENT_FOLDER, filename)
    cv2.imwrite(path, frame)


# ==========================
# LANE DETECTION
# ==========================

def detect_lane(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height = edges.shape[0]
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (frame.shape[1], height),
        (frame.shape[1], int(height * 0.6)),
        (0, int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    cropped = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        cropped,
        2,
        np.pi / 180,
        100,
        minLineLength=50,
        maxLineGap=50
    )

    return lines


# ==========================
# COLLISION WARNING
# ==========================

def collision_warning(box_height, frame_height):

    ratio = box_height / frame_height

    if ratio > 0.40:
        return "HIGH"
    elif ratio > 0.25:
        return "MEDIUM"

    return "SAFE"


# ==========================
# AI DECISION ENGINE
# ==========================

def decision(persons, vehicles, traffic, signal):

    warnings = []

    if persons > 0:
        warnings.append("Pedestrian Ahead")

    if traffic == "HIGH":
        warnings.append("Heavy Traffic")

    if signal == "RED":
        warnings.append("Stop Vehicle")

    if vehicles > 5:
        warnings.append("Maintain Safe Distance")

    return warnings


# ==========================
# AI ENGINE
# ==========================

class DriveEasyAI:

    def __init__(self):

        print("Loading YOLO model...")

        self.net = cv2.dnn.readNet(
            "models/yolov4-tiny.weights",
            "models/yolov4-tiny.cfg"
        )

        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.layer_names = self.net.getLayerNames()

        self.output_layers = [
            self.layer_names[i - 1]
            for i in self.net.getUnconnectedOutLayers()
        ]

        self.signal_color = None

        print("AI Ready")

    def detect_signal_color(self, roi):

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))

        red = cv2.bitwise_or(red1, red2)

        yellow = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        green = cv2.inRange(hsv, (40, 50, 50), (90, 255, 255))

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

        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            (320, 320),
            swapRB=True
        )

        self.net.setInput(blob)

        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.35:

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.35, 0.4)

        persons = 0
        vehicles = 0
        detections = []

        if len(indices) > 0:

            for i in indices.flatten():

                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]

                detections.append((x, y, w, h, label))

                if label == "person":
                    persons += 1

                if label in ["car", "truck", "bus", "motorbike"]:
                    vehicles += 1

                if label == "traffic light":

                    roi = frame[y:y+h, x:x+w]

                    color = self.detect_signal_color(roi)

                    if color:
                        self.signal_color = color

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

    def __init__(self, camera, ai, gemini_ai):

        super().__init__()

        self.camera = camera
        self.ai = ai
        self.gemini_ai = gemini_ai
        self.frame_count = 0
        self.cached = ([], 0, 0, "LOW", None)

    async def recv(self):

        pts, time_base = await self.next_timestamp()

        frame = self.camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.frame_count % 5 == 0:
            self.cached = self.ai.detect_objects(frame)

        detections, persons, vehicles, traffic, signal_color = self.cached

        height, width = frame.shape[:2]

        warnings = decision(persons, vehicles, traffic, signal_color)

        # Merge Gemini AI warnings with YOLO rule-based warnings
        gemini_warnings = self.gemini_ai.get_warnings()
        for gw in gemini_warnings:
            if gw not in warnings:
                warnings.append(gw)

        # Submit frame to Gemini every 90 frames (~3 sec at 30fps)
        if self.frame_count % 90 == 0:
            self.gemini_ai.submit_frame(frame)

        lane_lines = detect_lane(frame)

        if lane_lines is not None:
            for line in lane_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        for (x, y, w, h, label) in detections:

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(frame, label, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            risk = collision_warning(h, height)

            if risk == "HIGH":
                warnings.append("Collision Risk")
                save_event(frame)

        for i, w in enumerate(warnings):

            cv2.putText(
                frame,
                w,
                (30, 80 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            speak(w)

        now = datetime.now()

        cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 0), -1)

        cv2.putText(frame,
                    "DriveEasy Eye AI Dashcam",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        cv2.rectangle(frame, (0, height-50), (width, height), (0, 0, 0), -1)

        cv2.putText(frame,
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    (20, height-15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2)

        info = f"Persons:{persons} Vehicles:{vehicles} Traffic:{traffic}"

        if signal_color:
            info += f" Signal:{signal_color}"

        cv2.putText(frame,
                    info,
                    (width-600, height-15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2)

        self.frame_count += 1

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame


# ==========================
# WEBRTC
# ==========================

async def run_webrtc_sender():

    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (1280, 720)}
    )

    picam2.configure(config)

    picam2.start()

    time.sleep(2)

    ai = DriveEasyAI()

    # Initialize Gemini free AI and start background thread
    gemini_ai = GeminiVisionAnalyzer()
    gemini_ai.start()

    async with websockets.connect(SIGNALING_SERVER) as ws:

        await ws.send(json.dumps({
            "type": "register-pi",
            "deviceId": DEVICE_ID
        }))

        async for message in ws:

            data = json.loads(message)

            if data["type"] == "request-offer":

                config = RTCConfiguration(
                    iceServers=[
                        RTCIceServer(
                            urls=["stun:stun.l.google.com:19302"]
                        )
                    ]
                )

                pc = RTCPeerConnection(configuration=config)

                video_track = AIVideoStreamTrack(picam2, ai, gemini_ai)

                pc.addTrack(video_track)

                offer = await pc.createOffer()

                await pc.setLocalDescription(offer)

                await ws.send(json.dumps({
                    "type": "offer",
                    "deviceId": DEVICE_ID,
                    "sdp": {
                        "type": pc.localDescription.type,
                        "sdp": pc.localDescription.sdp
                    }
                }))

            elif data["type"] == "answer":

                answer = RTCSessionDescription(
                    sdp=data["sdp"]["sdp"],
                    type=data["sdp"]["type"]
                )

                await pc.setRemoteDescription(answer)


if __name__ == "__main__":
    asyncio.run(run_webrtc_sender())
