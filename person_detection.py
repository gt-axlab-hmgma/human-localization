import cv2
from ultralytics import YOLO

# Load a small YOLO model
model = YOLO("yolo11n.pt")

# Open your webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    raise SystemExit

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    # Run YOLO on the current frame
    results = model(frame, verbose=False)

    for result in results:
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

            # Draw the box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw a red dot at the person's feet
            cv2.circle(
                frame,
                (person_x, person_y),
                8,
                (0, 0, 255),
                -1
            )

            # Show the pixel position
            cv2.putText(
                frame,
                f"Position: ({person_x}, {person_y})",
                (person_x + 10, person_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            # Label it
            cv2.putText(
                frame,
                f"Person {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()