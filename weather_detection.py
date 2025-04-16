import numpy as np
import cv2

class WeatherConditionDetector:
    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        brightness = np.mean(hsv[..., 2])

        if brightness < 50:
            return 'night'
        elif self.is_foggy(frame):
            return 'fog'
        elif self.is_rainy(frame):
            return 'rain'
        elif self.is_snowy(frame):
            return 'snow'
        else:
            return 'clear'
        
    def is_foggy(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < 100
    
    def is_rainy(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.sum(edges > 0) > 15000
    
    def is_snowy(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        white_pixels = np.sum(gray > 220)
        return white_pixels > 10000
    
    def apply_filter(self, frame, condition):
        if condition == 'fog':
            return cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        elif condition == 'rain' or condition == 'snow':
            return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        elif condition == 'night':
            return cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
        return frame
