import cv2
from detection import ObjectDetector
from speech import Speaker  # додали модуль озвучки
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

class VideoProcessor:
    def __init__(self, video_path, model_path):
        self.cap = cv2.VideoCapture(video_path)
        self.detector = ObjectDetector(model_path)
        self.speaker = Speaker()  # створили об’єкт озвучувача

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

        self.class_names = {0: "bus", 1: "tram", 2: "trolleybus", 3: "number"}
        self.last_spoken = ""  # щоб не озвучувати одне й те саме

        self.object_paths = {}
        self.seen_objects = {}
        self.spoken_objects = set()
        self.MIN_FRAMES_VISIBLE = 5

    def process_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.detector.detect(frame)

            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    class_id = int(cls)

                    if conf < 0.6:
                        continue

                    if class_id not in [0, 1, 2, 3]:
                        continue

                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    obj_id = f"{class_id}-{center[0]}-{center[1]}"

                    # Відстеження траєкторії
                    if obj_id not in self.object_paths:
                        self.object_paths[obj_id] = []
                    self.object_paths[obj_id].append(center)

                    for i in range(1, len(self.object_paths[obj_id])):
                        cv2.line(frame, self.object_paths[obj_id][i - 1], self.object_paths[obj_id][i], (0, 255, 0), 2)

                    label = self.class_names.get(class_id, "unknown")
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Підрахунок кадрів видимості
                    if obj_id not in self.seen_objects:
                        self.seen_objects[obj_id] = 1
                    else:
                        self.seen_objects[obj_id] += 1

                    if class_id == 3:
                        cropped = frame[int(y1):int(y2), int(x1):int(x2)]

                        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                        text = pytesseract.image_to_string(gray, config='--psm 7 -c tessedit_char_whitelist=0123456789')
                        number = ''.join(filter(str.isdigit, text))

                        print(number)

                        if number and obj_id not in self.spoken_objects:
                            self.speaker.say(f"Number {number}")
                            self.spoken_objects.add(obj_id)

                        cv2.putText(frame, f"#{number}", (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    elif self.seen_objects[obj_id] >= self.MIN_FRAMES_VISIBLE and obj_id not in self.spoken_objects:
                        self.speaker.say(f"{label} near")
                        self.spoken_objects.add(obj_id)

            cv2.imshow("Detection", frame)
            self.out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
