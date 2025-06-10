# Moves the mouse cursor based on hand gestures (index and thumb distance)
# Allows clicking with the index and middle fingers

import cv2
import numpy as np
import time
import autopy
import HandTrackingModule as htm  

class MouseControl:
    def __init__(self):
        self.cap = None
        self.streaming = False
        self.wCam, self.hCam = 640, 480
        self.frameR = 100  # Frame Reduction for movement box
        self.smoothening = 7  # Smoothing for cursor movement
        self.pTime = 0
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0

        # Initializing camera and screen
        self.detector = htm.handDetector(maxHands=1)
        self.wScr, self.hScr = autopy.screen.size()

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
            # Capturing frame
            success, img = self.cap.read()
            img = self.detector.findHands(img)
            lmList, bbox = self.detector.findPosition(img)

            # Only process if hand is detected
            if len(lmList) != 0:
                # Get finger states
                fingers = self.detector.fingersUp()

                # Draw movement rectangle
                cv2.rectangle(img, (self.frameR, self.frameR), (self.wCam - self.frameR, self.hCam - self.frameR),
                              (255, 0, 255), 2)

                # Moving Mode: Only index finger up
                if len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 0:
                    x1, y1 = lmList[8][1:]

                    # Convert to screen coordinates
                    x3 = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScr))
                    y3 = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScr))

                    # Smooth cursor movement
                    self.clocX = self.plocX + (x3 - self.plocX) / self.smoothening
                    self.clocY = self.plocY + (y3 - self.plocY) / self.smoothening

                    # Move mouse
                    autopy.mouse.move(self.wScr - self.clocX, self.clocY)
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    self.plocX, self.plocY = self.clocX, self.clocY

                # Clicking Mode: Index and middle fingers up
                elif len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 1:
                    length, img, lineInfo = self.detector.findDistance(8, 12, img)
                    if length < 40:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]),
                                   15, (0, 255, 0), cv2.FILLED)
                        autopy.mouse.click()

            # Display FPS
            cTime = time.time()
            fps = 1 / (cTime - self.pTime) if cTime != self.pTime else 0
            self.pTime = cTime
            cv2.putText(img, f"FPS: {int(fps)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the frame
            ret, buffer = cv2.imencode('.jpg', img)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        self.stop_camera()
