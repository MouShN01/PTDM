import config
import cv2
import time

from frame_grab import FrameGrabber
from speech import Speaker
from analyzation import Analizer
from weather_detection import WeatherConditionDetector

def main():
    
    video_path = config.VIDEO_PATH
    model_path = config.MODEL_PATH
    check_rate = config.CHECK_RATE

    frame_grabber = FrameGrabber(0, check_rate)
    fps = frame_grabber.get_fps()
    delay = 1.0 / fps if fps > 0 else 0.033
    speaker = Speaker()
    analyzer = Analizer(model_path)
    weather_detector = WeatherConditionDetector()

    frame, snapshot = frame_grabber.read()
    condition = weather_detector.detect(frame)

    
    while True:
        start = time.time()

        frame, snapshot = frame_grabber.read()
        if frame is None:
            break

        # показуємо відео завжди
        cv2.imshow("Detection", frame)

        #mod_snapshot = weather_detector.apply_filter(snapshot, condition)
        result = analyzer.analize(snapshot)
        if result:
            speaker.say(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed = time.time() - start
        if delay > elapsed:
            time.sleep(delay - elapsed)
        
    frame_grabber.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
