
import json
import os
import socket
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# BLE Anchor ID -> USB camera index
CAMERAS = {
    1: 0,
    3: 1
}

# Physical tracking area: 72 cm x 92 cm
FLOOR_WIDTH_M = 0.72
FLOOR_HEIGHT_M = 0.92

UDP_PORT = 5005
BLE_TIMEOUT_S = 3.0

CALIBRATION_FILE = "camera_calibration.json"
MODEL_PATH = "yolo11n.pt"

# Change to True if you want to redo calibration on startup
RECALIBRATE = False


# ============================================================
# BLE UDP RECEIVER
# ============================================================

ble_data = {}
ble_lock = threading.Lock()
ble_error = None


def receive_ble():
    global ble_error

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", UDP_PORT))
        sock.settimeout(1.0)

    except OSError as exc:
        ble_error = f"UDP {UDP_PORT} unavailable: {exc}"
        return

    print(f"BLE receiver listening on UDP port {UDP_PORT}")

    while True:
        try:
            raw, _ = sock.recvfrom(1024)

            parts = (
                raw.decode("utf-8", errors="replace")
                .strip()
                .split(",")
            )

            if len(parts) < 3 or parts[0] != "DATA":
                continue

            anchor = int(parts[1])

            if anchor not in CAMERAS:
                continue

            with ble_lock:

                if parts[2] == "LOST":
                    ble_data.pop(anchor, None)

                elif len(parts) >= 5:

                    rssi = float(parts[2])
                    distance = float(parts[3])
                    score = float(parts[4])

                    if not np.isfinite(
                        [rssi, distance, score]
                    ).all():
                        continue

                    ble_data[anchor] = {
                        "rssi": rssi,
                        "distance": distance,
                        "score": max(0.0, score),
                        "received": time.monotonic()
                    }

        except socket.timeout:
            pass

        except (ValueError, UnicodeError):
            continue

        except OSError as exc:
            ble_error = f"UDP receive failed: {exc}"
            break


def ble_snapshot():

    now = time.monotonic()

    with ble_lock:
        snapshot = {
            anchor: (
                dict(ble_data[anchor])
                if anchor in ble_data
                else None
            )
            for anchor in CAMERAS
        }

    for anchor, data in snapshot.items():

        if data is not None:
            data["age"] = now - data["received"]
            data["fresh"] = (
                data["age"] <= BLE_TIMEOUT_S
            )

    return snapshot


# ============================================================
# CAMERA CALIBRATION
# ============================================================

def calibrate(anchor, cap):

    points = []
    name = f"Calibrate Camera {anchor}"

    cv2.namedWindow(name)

    def mouse(event, x, y, flags, userdata):

        if (
            event == cv2.EVENT_LBUTTONDOWN
            and len(points) < 4
        ):
            points.append([x, y])

            print(
                f"Camera {anchor}, "
                f"corner {len(points)}: ({x}, {y})"
            )

    cv2.setMouseCallback(name, mouse)

    print(f"\nCalibrating Camera {anchor}")
    print("Click the SAME four physical floor corners:")
    print("1: Origin (0, 0)")
    print("2: X corner (72, 0) cm")
    print("3: Opposite corner (72, 92) cm")
    print("4: Y corner (0, 92) cm")
    print("ENTER = save | R = reset | ESC = cancel")

    while True:

        ok, frame = cap.read()

        if not ok:
            raise RuntimeError(
                f"Camera {anchor} read failed"
            )

        for i, point in enumerate(points):

            cv2.circle(
                frame,
                tuple(point),
                6,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                str(i + 1),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if len(points) >= 2:

            cv2.polylines(
                frame,
                [np.asarray(points, np.int32)],
                len(points) == 4,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Corners {len(points)}/4 | R reset | ENTER save",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            points.clear()

        elif key in (10, 13) and len(points) == 4:
            break

        elif key == 27:
            raise RuntimeError(
                "Calibration cancelled"
            )

    cv2.destroyWindow(name)

    src = np.float32(points)

    dst = np.float32([
        [0, 0],
        [FLOOR_WIDTH_M, 0],
        [FLOOR_WIDTH_M, FLOOR_HEIGHT_M],
        [0, FLOOR_HEIGHT_M]
    ])

    return cv2.getPerspectiveTransform(src, dst)


def calibration_settings():

    return {
        "width": FLOOR_WIDTH_M,
        "height": FLOOR_HEIGHT_M,
        "cameras": {
            str(k): v
            for k, v in CAMERAS.items()
        }
    }


def load_calibration():

    if RECALIBRATE or not os.path.isfile(
        CALIBRATION_FILE
    ):
        return {}

    try:

        with open(
            CALIBRATION_FILE,
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if data.get("settings") != calibration_settings():
            return {}

        return {
            int(k): np.asarray(v, dtype=np.float64)
            for k, v in data["matrices"].items()
        }

    except (OSError, ValueError, KeyError, TypeError):
        return {}


def save_calibration(matrices):

    data = {
        "settings": calibration_settings(),
        "matrices": {
            str(k): matrix.tolist()
            for k, matrix in matrices.items()
        }
    }

    with open(
        CALIBRATION_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=2)


# ============================================================
# YOLO11 PERSON DETECTION
# ============================================================

def detect(model, frame, matrix, anchor):

    result = model.predict(
        frame,
        classes=[0],
        conf=0.4,
        imgsz=640,
        verbose=False
    )[0]

    position = None

    if len(result.boxes):

        box = max(
            result.boxes,
            key=lambda b: float(b.conf[0])
        )

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].cpu().tolist()
        )

        # Approximate foot position
        foot = (
            (x1 + x2) // 2,
            y2
        )

        pixel = np.array(
            [[foot]],
            dtype=np.float32
        )

        world = cv2.perspectiveTransform(
            pixel,
            matrix
        )[0, 0]

        x, y = map(float, world)

        # Draw person bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 220, 0),
            2
        )

        # Draw estimated foot point
        cv2.circle(
            frame,
            foot,
            6,
            (0, 0, 255),
            -1
        )

        inside = (
            0 <= x <= FLOOR_WIDTH_M
            and 0 <= y <= FLOOR_HEIGHT_M
        )

        if inside:

            position = (x, y)

            label = (
                f"X {x * 100:.1f}cm  "
                f"Y {y * 100:.1f}cm"
            )

        else:
            label = "Outside calibrated floor area"

        cv2.putText(
            frame,
            label,
            (8, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

    cv2.putText(
        frame,
        f"CAMERA {anchor}",
        (8, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    return frame, position


# ============================================================
# BLE-WEIGHTED POSITION FUSION
# ============================================================

def fuse(positions, snapshot):

    valid = {
        anchor: position
        for anchor, position in positions.items()
        if position is not None
    }

    if not valid:
        return None, {}, "NO CAMERA POSITION"

    scores = {}

    for anchor in valid:

        data = snapshot[anchor]

        if data is not None and data["fresh"]:
            scores[anchor] = data["score"]
        else:
            scores[anchor] = 0.0

    total = sum(scores.values())

    # Do not silently use camera-only fusion
    if total <= 0:
        return None, {}, "NO LIVE BLE - FUSION PAUSED"

    weights = {
        anchor: score / total
        for anchor, score in scores.items()
    }

    x = sum(
        valid[anchor][0] * weights[anchor]
        for anchor in valid
    )

    y = sum(
        valid[anchor][1] * weights[anchor]
        for anchor in valid
    )

    return (
        (x, y),
        weights,
        "BLE FUSION ACTIVE"
    )


# ============================================================
# LIVE DASHBOARD
# ============================================================

def make_dashboard(
    position,
    positions,
    weights,
    snapshot,
    status
):

    image = np.full(
        (650, 570, 3),
        246,
        dtype=np.uint8
    )

    black = (35, 35, 35)
    gray = (110, 110, 110)

    cv2.putText(
        image,
        "CAMERA + BLE TRACKING",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        black,
        2
    )

    status_color = (
        (0, 130, 0)
        if status == "BLE FUSION ACTIVE"
        else (0, 0, 190)
    )

    cv2.putText(
        image,
        status,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        status_color,
        2
    )

    # Top-down map
    left = 135
    top = 95
    right = 423
    bottom = 463

    cv2.rectangle(
        image,
        (left, top),
        (right, bottom),
        black,
        2
    )

    # Grid
    for fraction in (0.25, 0.5, 0.75):

        gx = round(
            left + (right - left) * fraction
        )

        gy = round(
            top + (bottom - top) * fraction
        )

        cv2.line(
            image,
            (gx, top),
            (gx, bottom),
            (210, 210, 210),
            1
        )

        cv2.line(
            image,
            (left, gy),
            (right, gy),
            (210, 210, 210),
            1
        )

    cv2.putText(
        image,
        "72 cm (X)",
        (230, 488),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        black,
        1
    )

    cv2.putText(
        image,
        "92 cm (Y)",
        (15, 278),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        black,
        1
    )

    def pixel(xy):

        return (
            round(
                left
                + xy[0] / FLOOR_WIDTH_M
                * (right - left)
            ),
            round(
                bottom
                - xy[1] / FLOOR_HEIGHT_M
                * (bottom - top)
            )
        )

    # Show each camera's individual estimate
    camera_colors = {
        1: (200, 120, 0),
        3: (200, 0, 130)
    }

    for anchor, color in camera_colors.items():

        if positions.get(anchor) is not None:

            cv2.circle(
                image,
                pixel(positions[anchor]),
                6,
                color,
                -1
            )

    # Final BLE-weighted position
    if position is not None:

        cv2.circle(
            image,
            pixel(position),
            10,
            (0, 0, 255),
            -1
        )

        label = (
            f"FUSED: X={position[0] * 100:.1f} cm   "
            f"Y={position[1] * 100:.1f} cm"
        )

    else:
        label = "FUSED: unavailable"

    cv2.putText(
        image,
        label,
        (20, 518),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        black,
        2
    )

    # BLE table headers
    headers = [
        (20, "ANCHOR"),
        (125, "RSSI"),
        (210, "DIST"),
        (302, "SCORE"),
        (408, "WEIGHT")
    ]

    for x, label in headers:

        cv2.putText(
            image,
            label,
            (x, 551),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            gray,
            1
        )

    # Live anchor readings
    for i, anchor in enumerate(CAMERAS):

        row_y = 578 + i * 30

        data = snapshot[anchor]

        live = (
            data is not None
            and data["fresh"]
        )

        state = "LIVE" if live else "OFF"

        state_color = (
            (0, 125, 0)
            if live
            else (0, 0, 170)
        )

        cv2.putText(
            image,
            f"{anchor} {state}",
            (20, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            state_color,
            1
        )

        if live:

            rssi = f"{data['rssi']:.0f} dBm"
            distance = f"{data['distance']:.2f}m"
            score = f"{data['score']:.3f}"

        else:

            rssi = "--"
            distance = "--"
            score = "--"

        values = [
            (125, rssi),
            (210, distance),
            (302, score),
            (
                408,
                f"{weights.get(anchor, 0) * 100:.0f}%"
            )
        ]

        for x, value in values:

            cv2.putText(
                image,
                value,
                (x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                black,
                1
            )

    footer = (
        ble_error
        or "Q quit | R recalibrate | ESP32 UDP:5005"
    )

    cv2.putText(
        image,
        footer[:75],
        (12, 640),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        gray,
        1
    )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    # Start BLE receiver in background thread
    threading.Thread(
        target=receive_ble,
        daemon=True
    ).start()

    caps = {}

    try:

        # Open both cameras
        for anchor, index in CAMERAS.items():

            cap = cv2.VideoCapture(
                index,
                cv2.CAP_DSHOW
            )

            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                640
            )

            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                480
            )

            if not cap.isOpened():

                raise RuntimeError(
                    f"Cannot open USB camera index {index}"
                )

            caps[anchor] = cap

        # Load or create calibration
        matrices = load_calibration()

        for anchor, cap in caps.items():

            if anchor not in matrices:

                matrices[anchor] = calibrate(
                    anchor,
                    cap
                )

        save_calibration(matrices)

        # Load YOLO11
        print("Loading YOLO11...")

        model = YOLO(MODEL_PATH)

        print("Tracking started!")
        print("Q = quit")
        print("R = recalibrate both cameras")

        while True:

            positions = {}

            # Process both camera feeds
            for anchor, cap in caps.items():

                ok, frame = cap.read()

                if not ok:

                    positions[anchor] = None
                    continue

                frame, positions[anchor] = detect(
                    model,
                    frame,
                    matrices[anchor],
                    anchor
                )

                cv2.imshow(
                    f"Camera {anchor}",
                    frame
                )

            # Get live BLE measurements
            snapshot = ble_snapshot()

            # Calculate weighted X/Y
            fused, weights, status = fuse(
                positions,
                snapshot
            )

            # Display live dashboard
            dashboard = make_dashboard(
                fused,
                positions,
                weights,
                snapshot,
                status
            )

            cv2.imshow(
                "Top-Down + BLE Status",
                dashboard
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):

                for anchor, cap in caps.items():

                    matrices[anchor] = calibrate(
                        anchor,
                        cap
                    )

                save_calibration(matrices)

    finally:

        for cap in caps.values():
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
