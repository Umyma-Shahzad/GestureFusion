import cv2
import numpy as np

class FilterCamera:
    def __init__(self):
        self.cap = None
        self.streaming = False
        self.wCam, self.hCam = 640, 480
        self.filter_mode = "original"
        self.sharpen_kernel = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]])

    def start_camera(self):
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.streaming = True

    def stop_camera(self):
        self.streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.filter_mode = "original"

    def set_filter_mode(self, mode):
        self.filter_mode = mode

    def apply_filter(self, frame):
        if self.filter_mode == "gray":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.filter_mode == "blur":
            return cv2.GaussianBlur(frame, (15, 15), 0)
        elif self.filter_mode == "edges":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.Canny(gray, 50, 150)
        elif self.filter_mode == "sharpen":
            return cv2.filter2D(frame, -1, self.sharpen_kernel)
        else:
            return frame

    def generate(self):
        self.start_camera()
        while self.streaming:
            ret, frame = self.cap.read()
            if not ret:
                continue

            filtered = self.apply_filter(frame)
            if len(filtered.shape) == 2:
                filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

            ret, buffer = cv2.imencode('.jpg', filtered)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        self.stop_camera()
