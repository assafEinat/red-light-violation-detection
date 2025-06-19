import cv2
import numpy as np
import pytesseract
import re

# Regex: Match 7 or 8 digit numbers
LICENSE_PLATE_REGEX = r'\b\d{7,8}\b'

def format_plate_number(plate_number):
    """Formats the number with dashes"""
    if len(plate_number) == 8:
        return f"{plate_number[:3]}-{plate_number[3:5]}-{plate_number[5:]}"
    elif len(plate_number) == 7:
        return f"{plate_number[:2]}-{plate_number[2:5]}-{plate_number[5:]}"
    else:
        return plate_number  # fallback

def extract_license_plate_number(car_roi):
    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)

    # Find candidate rectangles using contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(cnt)

        # Filter invalid sizes/aspect ratios
        if w_plate < 50 or h_plate < 15 or w_plate / h_plate > 6 or w_plate / h_plate < 2:
            continue

        # Extract ROI from the car image
        roi = gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]

        # OCR: Allow only digits
        text = pytesseract.image_to_string(roi, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        text = text.strip().replace(" ", "").replace("-", "")

        # Match a valid license plate number
        if re.fullmatch(LICENSE_PLATE_REGEX, text):
            return format_plate_number(text)

    return None  # No valid plate found
