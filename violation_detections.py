import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from line_detector import LineDetector
import time

# Constants
LICENSE_PLATE_OFFSET = 0.2  # 20% from car bottom

class FrameBatchProcessor:
    def __init__(self, batch_size=60):
        self.batch_size = batch_size
        self.frames = []
        self.violation_data = []
        self.line_detector = LineDetector(num_frames_avg=batch_size)
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
        """Process a batch of frames in two passes:
        1. Detect average stop line
        2. Detect traffic light, vehicles, and violations
        """


        # Pass 1: Detect and average stop line
        resized_frames = []
        stop_line_coords = []

        resized_w, resized_h = 640, 480

        for frame in frames:
            # Resize for processing
            resized = cv2.resize(frame, (resized_w, resized_h))
            resized_frames.append(resized)

            # Detect line in resized frame
            (x1, y1), (x2, y2), channel_indices = self.line_detector.detect_white_line(resized, color='red')

            # Save line coordinates (in resized space)
            stop_line_coords.append((y1, x1, x2))


        # Final average stop line
        if stop_line_coords:
            ys, xs1, xs2 = zip(*stop_line_coords)
            avg_y = int(np.mean(ys))
            avg_x1 = int(np.mean(xs1))
            avg_x2 = int(np.mean(xs2))

            # Convert coordinates back to original frame scale
            orig_h, orig_w = frames[-1].shape[:2]
            scale_x = orig_w / resized_w
            scale_y = orig_h / resized_h

            orig_y = int(avg_y * scale_y)
            orig_x1 = int(avg_x1 * scale_x)
            orig_x2 = int(avg_x2 * scale_x)

            # Draw on the original frame (last one used)
            cv2.line(
                frames[-1],  # or whichever original frame you want to draw on
                (orig_x1, orig_y),
                (orig_x2, orig_y),
                (0, 255, 0), 2
            )

            # Save state if needed
            self.current_stop_line_y = orig_y
            self.current_stop_line_x1 = orig_x1
            self.current_stop_line_x2 = orig_x2
            stop_line_detected = True
        else:
            self.current_stop_line_y = None
            self.current_stop_line_x1 = None
            self.current_stop_line_x2 = None
            stop_line_detected = False


        # Pass 2: Detect vehicles, traffic light, and violations
        for frame in resized_frames:
            processed_frame = frame.copy()
            results = self.yolo_model(frame, verbose=False)[0]

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

                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                plate_y = b - int((b - t) * LICENSE_PLATE_OFFSET)

                if track_id not in self.car_states:
                    self.car_states[track_id] = {
                        "has_passed": False,
                        "initial_y": plate_y
                    }
                    continue

                if (
                        self.current_traffic_light_state == 'red' and
                        stop_line_detected and
                        not self.car_states[track_id]["has_passed"] and
                        self.car_states[track_id]["initial_y"] < self.current_stop_line_y <= plate_y and
                        self.current_stop_line_x1 <= car_center_x <= self.current_stop_line_x2
                    ):
                    
                    violation = {
                        "track_id": track_id,
                        "position": (l, t, r, b),
                        "violation": "Red light violation",
                        "points": 10,
                    }

                    self.car_states[track_id]["has_passed"] = True
                    self.last_violations[track_id] = time.time()

                    cv2.rectangle(processed_frame, (l, t), (r, b), (0, 0, 255), 2)
                    cv2.putText(processed_frame, f"VIOLATION (ID: {track_id})",
                                (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    self.violation_data.append((violation, processed_frame))

                car_center_x = (l + r) // 2
                cv2.circle(processed_frame, (car_center_x, plate_y), 5, (255, 0, 0), -1)

