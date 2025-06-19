import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from line_detector import Line
from extract_plate_number import extract_license_plate_number
import time


class FrameBatchProcessor:
    def __init__(self, position, batch_size=60):
        self.batch_size = batch_size
        self.frames = []
        self.violation_data = []
        self.line = Line(position)
        self.current_stop_line_y = None
        self.yolo_model = YOLO("yolov8s.pt")
        self.tracker = DeepSort(max_age=30)
        self.car_states = {}
        self.last_violations = {}
        self.current_traffic_light_state = None



    def add_frame(self, frame):
        self.frames.append(frame)
        if len(frame) >= 60:
            self.process_batch(self.frames)

            violations = [v[0] for v in self.violation_data]
            violation_frames = [v[1] for v in self.violation_data]

            self.frames = []
            self.violation_data = []
            self.last_violations = {}
            self.car_states = {}
            self.processed_frames = []
            return violations, violation_frames
        return [], []


    def detect_traffic_light_state(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 'unknown'
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Color ranges (HSV space)
        lower_red1 = np.array([0, 150, 150])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 150])  # Red wraps around in HSV
        upper_red2 = np.array([180, 255, 255])
        
        lower_yellow = np.array([20, 150, 150])
        upper_yellow = np.array([30, 255, 255])
        
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Count pixels
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Thresholds (adjust based on your needs)
        threshold = 50  # Minimum bright pixels to consider
        
        if red_pixels > threshold:
            return 'red'
        elif yellow_pixels > threshold:
            return 'yellow'
        elif green_pixels > threshold:
            return 'green'
        return 'unknown'

    def process_batch(self, frames):
        """Process a batch of frames in one passes:
        1. Detect traffic light, vehicles, and violations
        """

        resized_w, resized_h = 640, 480

        # Pass: Detect vehicles, traffic light, and violations
        for frame in frames:
            resized_frame = frame.copy()
            resized_frame = cv2.resize(resized_frame, (resized_w, resized_h))

            processed_frame = resized_frame.copy()
            results = self.yolo_model(frame, verbose=False)[0]
            processed_frame = self.line.draw_line(frame)
            detections = []
            traffic_lights = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.yolo_model.names[class_id]

                if label == 'car' and confidence > 0.3:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'car'))

                elif label == 'traffic light' and confidence > 0.3:
                    traffic_lights.append((x1, y1, x2, y2))
                    light_state = self.detect_traffic_light_state(frame, (x1, y1, x2, y2))
                    self.current_traffic_light_state = light_state

                    color = (0, 0, 255) if light_state == 'red' else (0, 255, 0)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(processed_frame, f"Traffic Light: {light_state}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Track vehicles
            tracks = self.tracker.update_tracks(detections, frame=processed_frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                l, t, r, b = map(int, track.to_ltrb())

                print(self.current_traffic_light_state == 'red')
                print(self.line)
                if (
                        self.current_traffic_light_state == 'red' and
                        self.line and
                        self.line.is_car_touch_line(l, t, r, processed_frame.shape)
                    ):
                    violation_box = (l, t, r, b)
                    car_crop = crop_car_from_frame(frame, violation_box) 
                    license_plate_number = extract_license_plate_number(car_crop)

                    if license_plate_number not in self.last_violations.keys():
                        self.last_violations[license_plate_number] = license_plate_number
                    else:
                        continue

                    violation = {
                        "license_plate_number": license_plate_number,
                        "violation": "Red light violation",
                        "points": 10,
                    }


                    cv2.rectangle(processed_frame, (l, t), (r, b), (0, 0, 255), 2)
                    cv2.putText(processed_frame, f"VIOLATION (ID: {license_plate_number})",
                                (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    self.violation_data.append((violation, processed_frame))



def crop_car_from_frame(frame, position, frame_resized_shape=(640, 480)):
    l, t, r, b = position
    resized_w, resized_h = frame_resized_shape
    orig_h, orig_w = frame.shape[:2]

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
    car_crop = frame[orig_t:orig_b, orig_l:orig_r].copy()
    return car_crop
