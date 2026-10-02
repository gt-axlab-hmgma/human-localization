
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    raise SystemExit

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    cv2.imshow("Brio 300 Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()