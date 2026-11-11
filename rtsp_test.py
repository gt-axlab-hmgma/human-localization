"""
Quick live view of every camera in CAMERAS (USB or RTSP).

    python rtsp_test.py          # all cameras
    python rtsp_test.py 3        # just anchor 3

Shows each feed scaled to fit the screen with its resolution and
frame rate. Q or ESC quits.
"""

import sys
import time

# Import the fusion module first: it sets the FFmpeg options
# (TCP transport, quiet logging) before cv2 loads.
from ble_camera_fusion import (
    CAMERAS,
    load_secrets,
    open_camera
)

import cv2

MAX_WIDTH = 960


def main():

    wanted = (
        [int(a) for a in sys.argv[1:]]
        if len(sys.argv) > 1
        else list(CAMERAS)
    )

    secrets = load_secrets()
    caps = {}

    for anchor in wanted:
        caps[anchor] = open_camera(
            anchor,
            CAMERAS[anchor],
            secrets
        )

    counts = {a: 0 for a in caps}
    fps = {a: 0.0 for a in caps}
    last = time.monotonic()

    print("Q or ESC to quit")

    try:
        while True:

            for anchor, cap in caps.items():

                ok, frame = cap.read()

                if not ok:
                    print(f"Camera {anchor}: no frame")
                    continue

                counts[anchor] += 1
                h, w = frame.shape[:2]

                if w > MAX_WIDTH:
                    scale = MAX_WIDTH / w
                    frame = cv2.resize(
                        frame,
                        (MAX_WIDTH, int(h * scale))
                    )

                cv2.putText(
                    frame,
                    f"Camera {anchor}  {w}x{h}  {fps[anchor]:.1f} fps",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(f"Camera {anchor}", frame)

            now = time.monotonic()

            if now - last >= 1.0:

                for anchor in caps:
                    fps[anchor] = counts[anchor] / (now - last)
                    counts[anchor] = 0

                last = now

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:
        for cap in caps.values():
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
