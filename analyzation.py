from detection import ObjectDetector
from class_names import class_names
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

class Analizer:
    def __init__(self, model):
        self.detector = ObjectDetector(model)

    def analize(self, snapshot):
        if snapshot is not None:
            results = self.detector.detect(snapshot)

            transports = []
            numbers = []

            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    if conf < 0.65:
                        continue
                    class_id = int(cls)
                    box_data = (int(x1),int(y1),int(x2),int(y2))

                    if class_id in [0, 1, 2]:
                        transports.append((class_id, box_data))
                    elif class_id == 3:
                        numbers.append(box_data)

            if numbers:
                for class_id, (tx1, ty1, tx2, ty2) in transports:
                    for nx1, ny1, nx2, ny2 in numbers:
                        if nx1 > tx1 and ny1 > ty1 and nx2 < tx2 and ny2 < ty2:
                            cropped = snapshot[int(ny1):int(ny2), int(nx1):int(nx2)]
                            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                            text = pytesseract.image_to_string(
                                gray, config='--psm 7 -c tessedit_char_whitelist=0123456789'
                            )
                            number = ''.join(filter(str.isdigit, text))
                            if number:
                                label = class_names.get(class_id, "unknown")
                                return f"{label} {number} is coming"

            # якщо номерів не було або не співпали — просто озвучити транспорт
            if transports:
                class_id, _ = transports[0]
                label = class_names.get(class_id, "unknown")
                return f"{label} is coming"

            return None
