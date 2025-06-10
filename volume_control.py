# the distance between the index finger and thumb is used to adjust the system's volume

import cv2
import time
import numpy as np
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class VolumeControl:
    def __init__(self):
        self.cap = None
        self.streaming = False
        self.wCam, self.hCam = 640, 480
        self.detector = htm.handDetector(detectionCon=0.7, maxHands=1)

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))

        volRange = self.volume.GetVolumeRange()
        self.minVol = volRange[0]
        self.maxVol = volRange[1]
        self.vol = 0
        self.volBar = 400
        self.volPer = 0
        self.colorVol = (255, 0, 0)
        self.pTime = 0

    def start_camera(self):
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.streaming = True

    def stop_camera(self):
        self.streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def generate(self):
        self.start_camera()
        while self.streaming:
            success, img = self.cap.read()

            # Finding Hand
            img = self.detector.findHands(img)
            lmList, bbox = self.detector.findPosition(img, draw=True)
            if len(lmList) != 0:

                # Filtering based on size
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
                if 250 < area < 1000:

                    # Find Distance between index and Thumb
                    length, img, lineInfo = self.detector.findDistance(4, 8, img)

                    # Convert Volume
                    self.volBar = np.interp(length, [50, 200], [400, 150])
                    self.volPer = np.interp(length, [50, 200], [0, 100])

                    # Reduce Resolution to make it smoother
                    smoothness = 10
                    self.volPer = smoothness * round(self.volPer / smoothness)

                    # If  down set volume
                    fingers = self.detector.fingersUp()
                    if not fingers[4]:
                        self.volume.SetMasterVolumeLevelScalar(self.volPer / 100, None)
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        self.colorVol = (0, 255, 0)
                    else:
                        self.colorVol = (255, 0, 0)

            # Drawings
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(self.volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(self.volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)
            cVol = int(self.volume.GetMasterVolumeLevelScalar() * 100)
            cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                        1, self.colorVol, 3)

            # Frame rate
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)

            ret, buffer = cv2.imencode('.jpg', img)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        self.stop_camera()
