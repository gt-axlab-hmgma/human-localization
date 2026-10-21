import cv2


def find_cameras(max_index=6):
    """Return a list of camera indices that open and deliver a frame."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(i)
        cap.release()
    return found


available = find_cameras()
print(f"Cameras found at indices: {available}")

if len(available) < 2:
    print("Need two working cameras, found", len(available))
    raise SystemExit

idx1, idx2 = available[0], available[1]

cap1 = cv2.VideoCapture(idx1, cv2.CAP_MSMF)
cap2 = cv2.VideoCapture(idx2, cv2.CAP_MSMF)

if not cap1.isOpened() or not cap2.isOpened():
    print("Could not open one of the cameras")
    raise SystemExit

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Could not read frame")
        break

    cv2.imshow(f"Camera {idx1}", frame1)
    cv2.imshow(f"Camera {idx2}", frame2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
