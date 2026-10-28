
import cv2

def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

cam1 = open_camera(0)
cam3 = open_camera(1)

if not cam1.isOpened():
    raise RuntimeError("Camera index 0 failed")

if not cam3.isOpened():
    raise RuntimeError("Camera index 1 failed")

print("Both cameras connected!")
print("Press Q to quit.")

try:
    while True:
        ret1, frame1 = cam1.read()
        ret3, frame3 = cam3.read()

        if not ret1 or not ret3:
            print("Failed to read a camera frame")
            break

        cv2.putText(
            frame1, "CAMERA 1 / ANCHOR 1",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            frame3, "CAMERA 3 / ANCHOR 3",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.imshow("Camera 1", frame1)
        cv2.imshow("Camera 3", frame3)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cam1.release()
    cam3.release()
    cv2.destroyAllWindows()
