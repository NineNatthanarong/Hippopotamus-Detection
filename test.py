import cv2
import numpy as np
from ultralytics import YOLO

# --- Tracking parameters ---
MAX_TRACKED_OBJECTS = 2
MAX_LOST_FRAMES = 100  # How long to keep "lost" objects

class TrackedObject:
    def __init__(self, obj_id, centroid, bbox):
        self.id = obj_id
        self.centroid = centroid
        self.bbox = bbox
        self.lost_frames = 0
        self.trace = [centroid]  # Store centroid history

def get_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def iou(boxA, boxB):
    # Compute intersection over union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# Load YOLO model
model = YOLO("best.pt")

video_path = "short.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "output.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

next_id = 0
tracked_objects = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            bbox = list(map(int, box[:4]))
            centroid = get_centroid(bbox)
            detections.append((centroid, bbox))

    # --- Tracking logic ---
    updated_objects = []
    used_detections = set()
    for obj in tracked_objects:
        # If object is lost, increase search range
        search_range = 50 + obj.lost_frames * 10  # Expand search range for lost objects
        min_dist = float('inf')
        min_idx = -1
        for idx, (centroid, bbox) in enumerate(detections):
            if idx in used_detections:
                continue
            # Avoid overlap with other tracked objects
            overlap = False
            for other in tracked_objects:
                if other is not obj and iou(bbox, other.bbox) > 0.2:
                    overlap = True
                    break
            if overlap:
                continue
            dist = np.linalg.norm(np.array(obj.centroid) - np.array(centroid))
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
        if min_idx != -1 and min_dist < search_range:
            # Update object
            obj.centroid, obj.bbox = detections[min_idx]
            obj.lost_frames = 0
            used_detections.add(min_idx)
            obj.trace.append(obj.centroid)  # Add new centroid to trace
        else:
            obj.lost_frames += 1
        updated_objects.append(obj)

    # Add new objects for unmatched detections (limit total tracked objects)
    for idx, (centroid, bbox) in enumerate(detections):
        if idx in used_detections:
            continue
        if len(updated_objects) < MAX_TRACKED_OBJECTS:
            updated_objects.append(TrackedObject(next_id, centroid, bbox))
            next_id += 1

    tracked_objects = updated_objects

    # Draw tracked objects and tracking lines
    overlay = frame.copy()
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox
        color = (0, 255, 0) if obj.lost_frames == 0 else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID {obj.id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # Draw tracking line with opacity
        if len(obj.trace) > 1:
            pts = np.array(obj.trace, dtype=np.int32)
            cv2.polylines(overlay, [pts], False, (255, 0, 0), 3)
    # Blend overlay with frame for opacity effect
    alpha = 0.5  # Opacity of the tracking line
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    out.write(frame)

cap.release()
out.release()