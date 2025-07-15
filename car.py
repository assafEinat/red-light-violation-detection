import cv2
import pytesseract
import re

# Updated Regex: Match plates with alphanumeric characters (min 6 chars)
LICENSE_PLATE_REGEX = r'^[0-9]{6,8}$'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class Car:
    def __init__(self, position, frame):
        self.position = position
        self.frame = frame


    def crop_car_from_frame(self,frame_resized_shape=(640, 480)):
        l, t, r, b = self.position
        resized_w, resized_h = frame_resized_shape
        orig_h, orig_w = self.frame.shape[:2]

        # Scale from resized to original
        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h

        # Convert to original coordinates
        orig_l = int(l * scale_x)
        orig_t = int(t * scale_y)
        orig_r = int(r * scale_x)
        orig_b = int(b * scale_y)

        # Ensure coordinates are within frame bounds
        orig_l = max(0, orig_l)
        orig_t = max(0, orig_t)
        orig_r = min(orig_w, orig_r)
        orig_b = min(orig_h, orig_b)

        # Crop and return the car image
        car_crop = self.frame[orig_t:orig_b, orig_l:orig_r].copy()
        return car_crop


    def format_plate_number(plate_number):
        """Formats the number with dashes (basic logic)"""
        plate_number = plate_number.upper()
        if len(plate_number) == 8:
            return f"{plate_number[:3]}-{plate_number[3:5]}-{plate_number[5:]}"
        elif len(plate_number) == 7:
            return f"{plate_number[:2]}-{plate_number[2:5]}-{plate_number[5:]}"
        else:
            return plate_number  # fallback


    def extract_license_plate_number(self):
        car_roi = self.crop_car_from_frame()

        if car_roi is None or car_roi.size == 0:
            print("[!] Warning: car_roi is empty, skipping.")
            return None

        # For visualization
        visual = car_roi.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Morphological closing to join characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Edge detection
        edges = cv2.Canny(closed, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 50 or h < 15 or w / h > 6 or w / h < 2:
                continue

            # Draw the rectangle
            cv2.rectangle(visual, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi = gray[y:y+h, x:x+w]
            thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 15, 10)
            # OCR
            config = '--psm 6 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(thresh, config=config)
            text = ''.join(filter(str.isalnum, text)).upper()

            if re.fullmatch(LICENSE_PLATE_REGEX, text):
                plate_number = self.format_plate_number(text)
                print("[+] Plate detected:", plate_number)

                # Show the image with the detected contour
                
                return plate_number


        return None