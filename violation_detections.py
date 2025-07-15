import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from line import Line
from car import Car


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

        # HSV color ranges
        lower_red1 = np.array([0, 150, 150])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 150])
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

        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)

        threshold = 50
        margin = 1.3  # ratio difference to be confident

        # Pick the most dominant color if it's confidently stronger
        if red_pixels > threshold and red_pixels > yellow_pixels * margin and red_pixels > green_pixels * margin:
            return 'red'
        elif yellow_pixels > threshold and yellow_pixels > red_pixels * margin and yellow_pixels > green_pixels * margin:
            return 'yellow'
        elif green_pixels > threshold and green_pixels > red_pixels * margin and green_pixels > yellow_pixels * margin:
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
                cv2.line(processed_frame, (l, b), (r, b), (255, 255, 0), 2)
                if (
                        self.current_traffic_light_state == 'red' and
                        self.line.is_car_touch_line(l, b, r, processed_frame.shape)
                    ):
                    car = Car((l, t, r, b), processed_frame)
                    license_plate_number = car.extract_license_plate_number()

                    if license_plate_number in self.last_violations:
                        continue
                    self.last_violations[license_plate_number] = license_plate_number

                    violation = {
                        "license_plate_number": license_plate_number,
                        "violation": "Red light violation",
                        "points": 10,
                    }


                    cv2.rectangle(processed_frame, (l, t), (r, b), (0, 0, 255), 2)
                    cv2.putText(processed_frame, f"VIOLATION (ID: {license_plate_number})",
                                (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    self.violation_data.append((violation, processed_frame))



