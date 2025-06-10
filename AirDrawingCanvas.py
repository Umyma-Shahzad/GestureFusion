# Draw in the air when the index finger is raised.
# Erase on the canvas when both the index and middle fingers are raised.

import cv2
import numpy as np
import mediapipe as mp
import datetime
import os

class AirDrawingCanvas:
    def __init__(self):
        self.cap = None
        self.canvas = None
        self.wCam, self.hCam = 640, 480
        self.prev_x, self.prev_y = 0, 0
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.draw_util = mp.solutions.drawing_utils
        self.streaming = False  # control flag

    def start_camera(self):
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.streaming = True

    def stop_camera(self):
        self.streaming = False
        if self.cap:
              self.cap.release()
              self.cap = None
        # Clear state
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0


    def process_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            self.draw_util.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

            index_tip = hand.landmark[8]
            index_bottom = hand.landmark[6]
            middle_tip = hand.landmark[12]
            middle_bottom = hand.landmark[10]

            x, y = int(index_tip.x * w), int(index_tip.y * h)
            index_up = index_tip.y < index_bottom.y
            middle_up = middle_tip.y < middle_bottom.y

            if index_up and not middle_up:
                if self.prev_x == 0 and self.prev_y == 0:
                    self.prev_x, self.prev_y = x, y
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), (255, 0, 0), 5)
                self.prev_x, self.prev_y = x, y
            elif index_up and middle_up:
                cv2.circle(self.canvas, (x, y), 30, (0, 0, 0), -1)
                self.prev_x, self.prev_y = 0, 0
            else:
                self.prev_x, self.prev_y = 0, 0

        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        drawing_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        combined = cv2.add(frame_bg, drawing_fg)

        return combined

    def generate(self):
        self.start_camera()
        while self.streaming:
            frame = self.process_frame()
            if frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        self.stop_camera()  # release webcam when loop ends

    def save_canvas(self):
        if self.canvas is not None:
            filename = datetime.datetime.now().strftime("saved_drawings/drawing_%Y%m%d_%H%M%S.png")
            cv2.imwrite(filename, self.canvas)
            return filename
        return None
