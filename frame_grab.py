# frame_grab.py
import cv2
import time

class FrameGrabber:
    def __init__(self, source, interval_sec=1):
        self.cap = cv2.VideoCapture(source)
        self.interval = interval_sec
        self.last_time = time.time()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        snapshot = None
        current_time = time.time()
        if current_time - self.last_time >= self.interval:
            self.last_time = current_time
            snapshot = frame.copy()

        return frame, snapshot

    def get_fps(self):
        return self.fps

    def release(self):
        self.cap.release()
