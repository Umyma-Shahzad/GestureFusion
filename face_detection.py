import cv2

class FaceDetection:
    def __init__(self):
        self.cap = None
        self.streaming = False
        self.wCam, self.hCam = 640, 480
        # Loading Haar cascades for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

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
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detecting faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Drawing rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Geting the face region of interest for eye detection
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                # Detecting eyes within face region
                eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)

                for (ex, ey, ew, eh) in eyes:
                    # Drawing rectangle around eyes inside the face region
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        self.stop_camera()
