import cv2
from ultralytics import YOLO

# Load a small YOLO model (shared by both cameras)
model = YOLO("yolo11n.pt")

# Camera indices: 0 = Brio 300, 1 = USB Live camera
CAMERA_IDS = [0, 1]


def draw_people(frame, result):
    """Draw boxes, foot positions and labels for every person in a result."""
    people = []

    for box in result.boxes:
        class_id = int(box.cls[0])

        # COCO class 0 = person
        if class_id != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Estimate where the person is standing
        person_x = (x1 + x2) // 2
        person_y = y2
        confidence = float(box.conf[0])
        people.append((person_x, person_y, confidence))

        # Draw the box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw a red dot at the person's feet
        cv2.circle(frame, (person_x, person_y), 8, (0, 0, 255), -1)

        # Show the pixel position
        cv2.putText(
            frame,
            f"Position: ({person_x}, {person_y})",
            (person_x + 10, person_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        # Label it
        cv2.putText(
            frame,
            f"Person {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return people


# Open both cameras
caps = {}
for cam_id in CAMERA_IDS:
    cap = cv2.VideoCapture(cam_id, cv2.CAP_MSMF)    
    if not cap.isOpened():
        print(f"Could not open camera {cam_id}")
        raise SystemExit
    caps[cam_id] = cap

consecutive_failures = 0

while True:
    frames = {}
    for cam_id, cap in caps.items():
        ret, frame = cap.read()
        if ret:
            frames[cam_id] = frame

    # The Brio can return empty frames for a moment after opening.
    # Skip the iteration unless a camera has been dead for a while.
    if len(frames) < len(caps):
        consecutive_failures += 1
        if consecutive_failures > 100:
            missing = [c for c in caps if c not in frames]
            print(f"Camera(s) {missing} stopped delivering frames")
            break
        continue
    consecutive_failures = 0

    # Run YOLO on both frames in one batched call
    results = model(list(frames.values()), verbose=False)

    for (cam_id, frame), result in zip(frames.items(), results):
        people = draw_people(frame, result)
        cv2.imshow(f"Person Detection - Camera {cam_id}", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
