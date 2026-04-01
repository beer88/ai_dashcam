#!/usr/bin/env python3

import asyncio
import cv2
from picamera2 import Picamera2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc import RTCConfiguration, RTCIceServer
from av import VideoFrame
import websockets
import json
import time
import os

# ================= CONFIG =================
SIGNALING_SERVER = "wss://driveeasy-eye-signaling-904892438797.us-central1.run.app/"
DEVICE_ID = "AICAM1"

# ================= VIDEO TRACK =================
class CameraStreamTrack(VideoStreamTrack):

    def __init__(self, cam):
        super().__init__()
        self.cam = cam

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        # Capture frame
        frame = self.cam.capture_array()

        # Convert format
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # OPTIONAL: Resize for smoothness (uncomment if needed)
        # frame = cv2.resize(frame, (640, 480))

        # Add simple header (optional)
        cv2.putText(frame, "DriveEasy Live Feed",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame

# ================= MAIN =================
async def run():

    # 🔥 Prevent camera lock
    os.system("sudo pkill -f libcamera")
    time.sleep(2)

    # Initialize camera
    cam = Picamera2()

    config = cam.create_video_configuration(
        main={"size": (1280, 720)}   # 🔥 HD smooth video
    )

    cam.configure(config)
    cam.start()
    time.sleep(2)

    print("Camera started successfully")

    # Connect signaling server
    async with websockets.connect(SIGNALING_SERVER) as ws:

        await ws.send(json.dumps({
            "type": "register-pi",
            "deviceId": DEVICE_ID
        }))

        async for message in ws:

            data = json.loads(message)

            if data["type"] == "request-offer":

                print("Received request-offer")

                pc = RTCPeerConnection(
                    RTCConfiguration(
                        iceServers=[
                            RTCIceServer(urls=["stun:stun.l.google.com:19302"])
                        ]
                    )
                )

                pc.addTrack(CameraStreamTrack(cam))

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

                print("Received answer")

                await pc.setRemoteDescription(
                    RTCSessionDescription(
                        sdp=data["sdp"]["sdp"],
                        type=data["sdp"]["type"]
                    )
                )

if __name__ == "__main__":
    asyncio.run(run())
