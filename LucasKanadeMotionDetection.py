# track and visualize the motion of feature points between frames from a webcam stream
# It performs:
# Shi-Tomasi feature detection: detects good corners in the first frame.
# Lucas-Kanade optical flow: tracks movement of those corners across video frames.
# Visual overlay: draws motion paths as lines and circles on a mask, then overlays it on the video.
# Live visualization: displays the optical flow in real time.
# Reinitialization: if tracking points fall below a threshold, it re-detects them.

import cv2
import numpy as np

class OpticalFlowVisualizer:
    def __init__(self):
        self.cap = None
        self.streaming = False
        self.wCam, self.hCam = 640, 480
        self.max_corners = 200
        self.quality_level = 0.01
        self.min_distance = 10
        self.prev_gray = None
        self.prev_points = None
        self.mask = None
        self.color = np.random.randint(0, 255, (self.max_corners, 3))

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.streaming = True

    def stop_camera(self):
        self.streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None
        # Clear state
        self.prev_gray = None
        self.prev_points = None
        self.mask = None
        self.color = np.random.randint(0, 255, (self.max_corners, 3))


    def generate(self):
        self.start_camera()
        ret, prev_frame = self.cap.read()
        if not ret:
            return

        self.prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, self.max_corners, self.quality_level, self.min_distance)
        self.mask = np.zeros_like(prev_frame)

        while self.streaming:
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None)

            if curr_points is not None and self.prev_points is not None:
                good_new = curr_points[status == 1]
                good_prev = self.prev_points[status == 1]

                for i, (new_pt, prev_pt) in enumerate(zip(good_new, good_prev)):
                    x_new, y_new = new_pt.ravel()
                    x_prev, y_prev = prev_pt.ravel()
                    dx = x_new - x_prev
                    dy = y_new - y_prev
                    motion_magnitude = np.sqrt(dx**2 + dy**2)

                    # Draw only if motion is significant
                    if motion_magnitude > 2:
                        cv2.arrowedLine(self.mask,
                                        (int(x_prev), int(y_prev)),
                                        (int(x_new), int(y_new)),
                                        self.color[i].tolist(),
                                        2,
                                        tipLength=0.4)

                output = cv2.add(frame, self.mask)
            else:
                output = frame.copy()

            ret, buffer = cv2.imencode('.jpg', output)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            self.prev_gray = gray
            self.prev_points = (
                good_new.reshape(-1, 1, 2)
                if curr_points is not None and len(curr_points) >= 50
                else cv2.goodFeaturesToTrack(self.prev_gray, self.max_corners, self.quality_level, self.min_distance)
            )

            if curr_points is None or len(curr_points) < 50:
                self.mask = np.zeros_like(frame)

        self.stop_camera()
