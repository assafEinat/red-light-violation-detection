import cv2
from ultralytics import YOLO
from pymongo import MongoClient
import easyocr
import time

# Load YOLOv8 model (download weights if not present)
model = YOLO("yolov8n.pt")  # Use yolov8n.pt for speed, or yolov8m.pt for better accuracy

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["traffic_monitor"]
violations_col = db["violations"]

# License plate reader
reader = easyocr.Reader(['en'])

# Load video
cap = cv2.VideoCapture("traffic.mp4")

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    results = model(frame)[0]

    for box in results.boxes:
        cls = model.names[int(box.cls[0])]
        if cls == "car" or cls == "truck":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Basic violation detection: example - crossing a line
            if y2 > 400:  # Violation line
                cv2.putText(frame, "VIOLATION", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                crop = frame[y1:y2, x1:x2]

                # OCR for license plate
                ocr_result = reader.readtext(crop)
                license_plate = ocr_result[0][1] if ocr_result else "Unknown"

                # Store in MongoDB
                violation = {
                    "frame_id": frame_id,
                    "timestamp": time.time(),
                    "plate": license_plate,
                    "violation": "Crossed Line",
                    "bbox": [x1, y1, x2, y2]
                }
                violations_col.insert_one(violation)

    cv2.line(frame, (0, 400), (frame.shape[1], 400), (0, 0, 255), 2)
    cv2.imshow("Traffic Violation Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
