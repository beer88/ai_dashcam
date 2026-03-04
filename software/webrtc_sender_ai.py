#!/usr/bin/env python3
"""
DriveEasy Eye - WebRTC Sender with AI Detection
Runs on Raspberry Pi, streams video with AI annotations to web dashboard
"""

import asyncio
import cv2
import numpy as np
from picamera2 import Picamera2
import json
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
from aiortc import RTCConfiguration, RTCIceServer
from av import VideoFrame
import time
from datetime import datetime
from aiortc import RTCIceCandidate

# Configuration
SIGNALING_SERVER = "wss://driveeasy-eye-signaling-904892438797.us-central1.run.app/"  # Update this after deploying
DEVICE_ID = "AICAM1"

class DriveEasyAI:
    """AI detection engine for DriveEasy Eye"""
    
    def __init__(self):
        print("Loading AI models...")
        
        # Load YOLO
        self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Traffic signal state
        self.traffic_light_state = None
        self.red_light_start_time = None
        self.avg_red_light_duration = 60
        
        print("AI models loaded successfully")
    
    def detect_traffic_light_color(self, roi):
        """Detect traffic light color"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Red
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Yellow
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        
        # Green
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
        
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        
        if max_pixels < 50:
            return None
        
        if red_pixels == max_pixels:
            return "RED"
        elif yellow_pixels == max_pixels:
            return "YELLOW"
        elif green_pixels == max_pixels:
            return "GREEN"
        
        return None
    
    def detect_objects(self, frame):
        """Run YOLO detection"""
        height, width, _ = frame.shape
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
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
                
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        
        detections = []
        traffic_light_roi = None
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                
                detections.append({
                    "class": label,
                    "confidence": round(confidence, 2),
                    "bbox": [x, y, w, h]
                })
                
                if label == "traffic light":
                    roi_x = max(0, x)
                    roi_y = max(0, y)
                    roi_w = min(w, width - roi_x)
                    roi_h = min(h, height - roi_y)
                    
                    if roi_w > 10 and roi_h > 10:
                        traffic_light_roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        return detections, traffic_light_roi
    
    def annotate_frame(self, frame, detections, traffic_light_color=None, wait_time=0):
        """Draw annotations on frame"""
        annotated = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection["bbox"]
            label = detection["class"]
            confidence = detection["confidence"]
            
            if label == "traffic light":
                color = (0, 0, 255)
            elif label in ["car", "truck", "bus"]:
                color = (255, 0, 0)
            elif label == "person":
                color = (0, 255, 0)
            else:
                color = (255, 255, 0)
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, f"{label} {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Traffic light info
        if traffic_light_color:
            info_text = f"Signal: {traffic_light_color}"
            if wait_time > 0:
                info_text += f" | Wait: {wait_time}s"
            
            cv2.rectangle(annotated, (10, 10), (400, 50), (0, 0, 0), -1)
            cv2.putText(annotated, info_text, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated

class AIVideoStreamTrack(VideoStreamTrack):
    """Video track that processes frames with AI"""
    
    def __init__(self, camera, ai_engine):
        super().__init__()
        self.camera = camera
        self.ai = ai_engine
        self.frame_count = 0
        
    async def recv(self):
        """Get next video frame with AI annotations"""
        pts, time_base = await self.next_timestamp()
        
        # Capture frame
        frame = self.camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run AI detection every 3rd frame to maintain performance
        if self.frame_count % 3 == 0:
            detections, traffic_light_roi = self.ai.detect_objects(frame)
            
            traffic_light_color = None
            wait_time = 0
            
            if traffic_light_roi is not None:
                traffic_light_color = self.ai.detect_traffic_light_color(traffic_light_roi)
                
                if traffic_light_color == "RED":
                    if self.ai.traffic_light_state != "RED":
                        self.ai.red_light_start_time = time.time()
                    self.ai.traffic_light_state = "RED"
                    
                    if self.ai.red_light_start_time:
                        elapsed = time.time() - self.ai.red_light_start_time
                        wait_time = max(0, int(self.ai.avg_red_light_duration - elapsed))
                else:
                    self.ai.traffic_light_state = traffic_light_color
                    self.ai.red_light_start_time = None
            
            # Annotate frame
            frame = self.ai.annotate_frame(frame, detections, traffic_light_color, wait_time)
        
        self.frame_count += 1
        
        # Convert to VideoFrame for WebRTC
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        
        return video_frame

async def run_webrtc_sender():
    """Main WebRTC sender with signaling"""
    
    print(f"Connecting to signaling server: {SIGNALING_SERVER}")
    
    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1080, 720)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("Camera ready")
    
    # Initialize AI
    ai = DriveEasyAI()
    
    # Connect to signaling server
    async with websockets.connect(SIGNALING_SERVER) as ws:
        print("Connected to signaling server")
        
        # Register as Pi device
        await ws.send(json.dumps({
            "type": "register-pi",
            "deviceId": DEVICE_ID
        }))
        
        response = await ws.recv()
        print(f"Registration response: {response}")
        
        pc = None
        
        async def handle_messages():
            nonlocal pc
            
            async for message in ws:
                data = json.loads(message)
                print(f"Received: {data['type']}")
                
                if data["type"] == "request-offer":
                    # Create new peer connection
                    config = RTCConfiguration(
                        iceServers=[
                            RTCIceServer(urls=["stun:stun.l.google.com:19302"])
                        ]
                    )

                    pc = RTCPeerConnection(configuration=config)

                    
                    # Add video track with AI
                    video_track = AIVideoStreamTrack(picam2, ai)
                    pc.addTrack(video_track)
                    
                    # Handle ICE candidates
                    @pc.on("icecandidate")
                    async def on_icecandidate(candidate):
                        if candidate:
                            await ws.send(json.dumps({
                                "type": "ice-candidate",
                                "deviceId": DEVICE_ID,
                                "candidate": {
                                    "candidate": candidate.candidate,
                                    "sdpMid": candidate.sdpMid,
                                    "sdpMLineIndex": candidate.sdpMLineIndex
                                }
                            }))
                    
                    # Create and send offer
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
                    
                    print("Offer sent")
                
                elif data["type"] == "answer":
                    # Receive answer from viewer
                    answer = RTCSessionDescription(
                        sdp=data["sdp"]["sdp"],
                        type=data["sdp"]["type"]
                    )
                    await pc.setRemoteDescription(answer)
                    print("Answer received, connection established")
                

                    from aiortc.sdp import candidate_from_sdp

                elif data["type"] == "ice-candidate":
                    if pc and data.get("candidate"):
                        cand = data["candidate"]

                        ice = candidate_from_sdp(cand["candidate"])
                        ice.sdpMid = cand.get("sdpMid")
                        ice.sdpMLineIndex = cand.get("sdpMLineIndex")

                    await pc.addIceCandidate(ice)

        
        # Keep connection alive
        await handle_messages()

if __name__ == "__main__":
    try:
        asyncio.run(run_webrtc_sender())
    except KeyboardInterrupt:
        print("\nStopping DriveEasy Eye sender...")
    except Exception as e:
        print(f"Error: {e}")


