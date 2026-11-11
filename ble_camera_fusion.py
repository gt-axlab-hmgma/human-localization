
import json
import os
from collections import deque
import socket
import threading
import time

# RTSP over TCP avoids the grey/torn frames UDP gives on Wi-Fi.
# Must be set before cv2 is imported.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp"
)

# Hide FFmpeg's "no frame!" spam while it waits for the first keyframe
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# BLE Anchor ID -> camera source.
#   int    = USB camera index
#   str    = RTSP URL. {user} and {password} are filled in from
#            camera_secrets.json (never put the password here).
CAMERAS = {
    # Find camera IPs with: python find_cameras.py
    1: "rtsp://{user}:{password}@172.20.10.2:10554/tcp/av0_0",   # app "camera 1", on the hotspot
    # 1: 0,   # USB webcam, if you want it back
    3: "rtsp://{user}:{password}@172.20.10.3:10554/tcp/av0_0"    # second camera, on the hotspot
}
# RTSP note: this camera model only accepts the lowercase path /tcp/av0_0

# BLE anchor ID -> where that ESP32 sits on the floor, in inches from
# corner 1 (same axes as the markers: X along the 210 in side, Y along
# the 190 in side). EVERY anchor that sends data must be listed here.
# An anchor with a camera uses the same ID in CAMERAS above; an anchor
# without a camera is "standalone" and only contributes BLE data.
# EDIT THESE to the real positions of your boards.
ANCHORS = {
    1: (0, 0),        # next to camera 1
    3: (210, 190)     # next to camera 3
    # 2: (210, 0)     # standalone anchor, no camera (add back when used)
}

# Username / password for the RTSP cameras, kept out of git
SECRETS_FILE = "camera_secrets.json"

# Physical tracking area, in INCHES. Corner 1 clicked during
# calibration is the origin (0, 0). X runs along the 210 in sides,
# Y along the 190 in sides, so the corners are:
#   1 = (0, 0)   2 = (210, 0)   3 = (210, 190)   4 = (0, 190)  [in]
FLOOR_WIDTH_IN = 210     # X side length
FLOOR_HEIGHT_IN = 190    # Y side length

# Internally everything is in meters (OpenCV transform, BLE distance)
M_PER_IN = 0.0254
FLOOR_WIDTH_M = FLOOR_WIDTH_IN * M_PER_IN
FLOOR_HEIGHT_M = FLOOR_HEIGHT_IN * M_PER_IN


def to_in(meters):

    return meters / M_PER_IN


def anchor_positions_m():

    return {
        anchor: (x * M_PER_IN, y * M_PER_IN)
        for anchor, (x, y) in ANCHORS.items()
    }

UDP_PORT = 5005
BLE_TIMEOUT_S = 3.0

CALIBRATION_FILE = "camera_calibration.json"
MODEL_PATH = "yolo11n.pt"

# Change to True if you want to redo calibration on startup
RECALIBRATE = False

# ---- ArUco auto-calibration ----
# Print the four markers with: python generate_markers.py
# Tape one at each floor corner. The CENTER of each marker is the
# corner, so FLOOR_WIDTH_IN / FLOOR_HEIGHT_IN are measured
# center-to-center between markers.
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Marker id -> corner index, same order as corner_labels():
#   corner 0 = (0,0)  1 = (W,0)  2 = (W,H)  3 = (0,H)
# Changing this invalidates camera_calibration.json.
ARUCO_CORNER_IDS = {0: 0, 1: 1, 2: 2, 3: 3}

ARUCO_AVG_FRAMES = 15     # frames averaged per marker center
ARUCO_STABLE_PX = 2.0     # max jitter (px) to count as stable

# Calibration window is shrunk to this width if the camera is bigger
CALIB_MAX_WIDTH = 1280

# How calibration starts: "manual" = click the corners by hand,
# "auto" = look for the ArUco markers. A and C switch modes live.
CALIBRATION_START_MODE = "manual"


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

            if anchor not in ANCHORS:
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
            for anchor in ANCHORS
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

def corner_labels():

    w = FLOOR_WIDTH_IN
    h = FLOOR_HEIGHT_IN

    return [
        "(0, 0)",
        f"({w}, 0)",
        f"({w}, {h})",
        f"(0, {h})"
    ]


def floor_corners_m():

    return np.float32([
        [0, 0],
        [FLOOR_WIDTH_M, 0],
        [FLOOR_WIDTH_M, FLOOR_HEIGHT_M],
        [0, FLOOR_HEIGHT_M]
    ])


def homography_from_corners(points):

    # Corner 1 -> (0, 0), then around the rectangle
    return cv2.getPerspectiveTransform(
        np.float32(points),
        floor_corners_m()
    )


_aruco_detector = None


def aruco_detector():

    global _aruco_detector

    if _aruco_detector is None:

        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX
        )

        _aruco_detector = cv2.aruco.ArucoDetector(
            dictionary,
            params
        )

    return _aruco_detector


def quad_center(quad):
    """
    Center of a marker as the intersection of its diagonals.
    Correct under perspective (unlike the mean of the corners)
    and independent of marker rotation.
    """

    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)

    p, r = q[0], q[2] - q[0]
    s, t = q[1], q[3] - q[1]

    denom = r[0] * t[1] - r[1] * t[0]

    if abs(denom) < 1e-9:
        c = q.mean(axis=0)
    else:
        u = ((s[0] - p[0]) * t[1] - (s[1] - p[1]) * t[0]) / denom
        c = p + u * r

    return (float(c[0]), float(c[1]))


def detect_markers(detector, frame):
    """
    Returns (found, corners, ids) where found maps corner index
    (0..3) -> (x, y) pixel center for every expected marker seen
    exactly once in this frame.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    found = {}

    if ids is not None:

        seen = {}

        for quad, marker_id in zip(corners, ids.ravel().tolist()):

            if marker_id in ARUCO_CORNER_IDS:
                seen.setdefault(marker_id, []).append(quad)

        for marker_id, quads in seen.items():

            # Same id twice (reflection, spare sheet) -> skip this frame
            if len(quads) == 1:
                found[ARUCO_CORNER_IDS[marker_id]] = quad_center(quads[0])

    return found, corners, ids


def update_history(history, found):

    for index, samples in history.items():

        if index in found:
            samples.append(found[index])

        elif samples:
            # Missed this frame: decay instead of resetting
            samples.popleft()


def averaged_corners(history):

    points = []
    jitter = 0.0

    for index in range(4):

        samples = history[index]

        if len(samples) < samples.maxlen:
            points.append(None)
            continue

        arr = np.asarray(samples, dtype=np.float64)
        mean = arr.mean(axis=0)

        points.append((float(mean[0]), float(mean[1])))
        jitter = max(jitter, float(arr.std(axis=0).max()))

    return points, jitter


def draw_corners(frame, points, labels):

    present = [
        (i, p) for i, p in enumerate(points)
        if p is not None
    ]

    for i, p in present:

        px = (int(round(p[0])), int(round(p[1])))

        cv2.circle(frame, px, 6, (0, 255, 0), -1)

        cv2.putText(
            frame,
            f"{i + 1} {labels[i]}",
            (px[0] + 8, px[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    if len(present) >= 2:

        cv2.polylines(
            frame,
            [np.asarray([p for _, p in present], np.int32)],
            len(present) == 4,
            (0, 255, 0),
            2
        )


def calibrate(anchor, cap):
    """
    Find the four floor corners in this camera.

    AUTO mode (default): looks for the four ArUco markers, averages
    their centers over ARUCO_AVG_FRAMES frames, ENTER saves once all
    four are stable.
    CLICK mode (press C): click the four corners by hand as before.

    Returns (homography, method) with method "aruco" or "manual".
    """

    name = f"Calibrate Camera {anchor}"
    labels = corner_labels()
    w_in = FLOOR_WIDTH_IN
    h_in = FLOOR_HEIGHT_IN

    state = {"mode": CALIBRATION_START_MODE}
    points = []                       # manual clicks
    history = {                       # auto: recent centers per corner
        i: deque(maxlen=ARUCO_AVG_FRAMES)
        for i in range(4)
    }
    detector = aruco_detector()

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    def mouse(event, x, y, flags, userdata):

        if (
            state["mode"] == "manual"
            and event == cv2.EVENT_LBUTTONDOWN
            and len(points) < 4
        ):
            points.append([x, y])

            print(
                f"Camera {anchor}, "
                f"corner {len(points)}: ({x}, {y})"
            )

    cv2.setMouseCallback(name, mouse)

    print(f"\nCalibrating Camera {anchor}")

    if state["mode"] == "manual":
        print("CLICK the 4 floor corners in order (A = switch to markers):")
        for i, label in enumerate(labels):
            print(f"  corner {i + 1} = {label} in")
        print(f"X side = {w_in} in, Y side = {h_in} in")
        print("ENTER = save | R = reset | ESC = cancel")

    else:
        print("AUTO: looking for the 4 ArUco markers at the floor corners")

        for marker_id, index in sorted(ARUCO_CORNER_IDS.items()):
            print(
                f"  marker {marker_id} = corner {index + 1} "
                f"{labels[index]} in"
            )

        print(
            f"X side = {w_in} in, Y side = {h_in} in "
            f"(center to center)"
        )
        print("ENTER = save when stable | C = click corners by hand")
        print("R = reset | ESC = cancel")

    sized = False
    yellow = (0, 255, 255)
    green = (0, 220, 0)

    while True:

        ok, frame = cap.read()

        if not ok:
            raise RuntimeError(
                f"Camera {anchor} read failed"
            )

        if not sized:

            h, w = frame.shape[:2]
            scale = min(1.0, CALIB_MAX_WIDTH / w)

            cv2.resizeWindow(
                name,
                int(w * scale),
                int(h * scale)
            )
            sized = True

        if state["mode"] == "auto":

            found, corners, ids = detect_markers(detector, frame)
            update_history(history, found)
            averaged, jitter = averaged_corners(history)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            draw_corners(frame, averaged, labels)

            missing = [
                str(i + 1) for i in range(4)
                if i not in found
            ]

            ready = (
                all(p is not None for p in averaged)
                and jitter <= ARUCO_STABLE_PX
            )

            if ready:

                status = (
                    f"AUTO 4/4 stable {jitter:.1f} px | "
                    f"ENTER save | C = click mode"
                )
                color = green

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (frame.shape[1] - 1, frame.shape[0] - 1),
                    green,
                    8
                )

            elif not missing:

                status = (
                    f"AUTO 4/4 | settling {jitter:.1f} px... | "
                    f"C = click mode"
                )
                color = yellow

            else:

                status = (
                    f"AUTO {len(found)}/4 markers | "
                    f"missing {','.join(missing)} | C = click mode"
                )
                color = yellow

            src = averaged

        else:

            padded = (
                [list(p) for p in points]
                + [None] * (4 - len(points))
            )

            draw_corners(frame, padded, labels)

            ready = len(points) == 4

            status = (
                f"CLICK {len(points)}/4 | R reset | "
                f"ENTER save | A = auto"
            )
            color = green if ready else yellow

            src = points

        cv2.putText(
            frame,
            status,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        cv2.putText(
            frame,
            f"1 = origin | X side {w_in} in | Y side {h_in} in",
            (8, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            yellow,
            2
        )

        cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c") and state["mode"] == "auto":

            state["mode"] = "manual"
            points.clear()
            print("Click mode: click the 4 corners in order")

        elif key == ord("a") and state["mode"] == "manual":

            state["mode"] = "auto"

            for samples in history.values():
                samples.clear()

            print("Auto mode: looking for markers")

        elif key == ord("r"):

            points.clear()

            for samples in history.values():
                samples.clear()

        elif key in (10, 13) and ready:
            break

        elif key == 27:
            raise RuntimeError(
                "Calibration cancelled"
            )

    cv2.destroyWindow(name)

    method = "aruco" if state["mode"] == "auto" else "manual"

    print(
        f"Camera {anchor}: {method} calibration, "
        f"corner 1 = origin, X side = {w_in} in, "
        f"Y side = {h_in} in"
    )

    for i, p in enumerate(src):
        print(
            f"  corner {i + 1} {labels[i]} in "
            f"-> pixel ({p[0]:.1f}, {p[1]:.1f})"
        )

    return homography_from_corners(src), method

def calibration_settings():

    return {
        "width": FLOOR_WIDTH_M,
        "height": FLOOR_HEIGHT_M,
        "cameras": {
            str(k): v
            for k, v in CAMERAS.items()
        },
        "aruco_dict": int(ARUCO_DICT),
        "marker_ids": {
            str(k): v
            for k, v in ARUCO_CORNER_IDS.items()
        }
    }


def load_calibration():

    if RECALIBRATE or not os.path.isfile(
        CALIBRATION_FILE
    ):
        return {}, {}

    try:

        with open(
            CALIBRATION_FILE,
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if data.get("settings") != calibration_settings():
            return {}, {}

        matrices = {
            int(k): np.asarray(v, dtype=np.float64)
            for k, v in data["matrices"].items()
        }

        methods = {
            int(k): str(v)
            for k, v in data.get("methods", {}).items()
        }

        for anchor in matrices:
            methods.setdefault(anchor, "manual")
            print(
                f"Camera {anchor} calibration: "
                f"{methods[anchor]} (from file)"
            )

        return matrices, methods

    except (OSError, ValueError, KeyError, TypeError):
        return {}, {}


def save_calibration(matrices, methods):

    data = {
        "settings": calibration_settings(),
        "methods": {
            str(k): v
            for k, v in methods.items()
        },
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
                f"X {to_in(x):.1f} in  "
                f"Y {to_in(y):.1f} in"
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

def ble_estimate(snapshot):
    """
    BLE-only position: proximity-score weighted centroid of the
    anchors that are live. Coarse (a few feet) but works with no
    camera at all.
    """

    anchors_m = anchor_positions_m()
    total = 0.0
    x = 0.0
    y = 0.0

    for anchor, data in snapshot.items():

        if data is None or not data["fresh"]:
            continue

        score = data["score"]

        if score <= 0:
            continue

        ax, ay = anchors_m[anchor]
        x += ax * score
        y += ay * score
        total += score

    if total <= 0:
        return None

    return (x / total, y / total)


def fuse(positions, snapshot):
    """
    Returns (position, weights, status, ble_only).
    position: camera positions weighted by their anchor's BLE score,
              or the BLE-only estimate when no camera sees the person.
    """

    ble_only = ble_estimate(snapshot)

    valid = {
        anchor: position
        for anchor, position in positions.items()
        if position is not None
    }

    if not valid:

        if ble_only is not None:
            return ble_only, {}, "BLE ONLY - NO CAMERA", ble_only

        return None, {}, "NO CAMERA POSITION", None

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
        return None, {}, "NO LIVE BLE - FUSION PAUSED", ble_only

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
        "BLE FUSION ACTIVE",
        ble_only
    )


# ============================================================
# LIVE DASHBOARD
# ============================================================

def make_dashboard(
    position,
    positions,
    weights,
    snapshot,
    status,
    ble_only=None
):

    # One table row per anchor; grow the panel beyond two anchors
    height = 650 + 30 * max(0, len(ANCHORS) - 2)

    image = np.full(
        (height, 570, 3),
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

    if status == "BLE FUSION ACTIVE":
        status_color = (0, 130, 0)
    elif status.startswith("BLE ONLY"):
        status_color = (0, 140, 230)
    else:
        status_color = (0, 0, 190)

    cv2.putText(
        image,
        status,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        status_color,
        2
    )

    # Top-down map, drawn to scale: X (210 in) across, Y (190 in) up,
    # corner 1 / origin at the bottom-left
    box_left, box_top = 110, 95
    box_right, box_bottom = 550, 463

    scale = min(
        (box_right - box_left) / FLOOR_WIDTH_M,
        (box_bottom - box_top) / FLOOR_HEIGHT_M
    )

    map_w = round(FLOOR_WIDTH_M * scale)
    map_h = round(FLOOR_HEIGHT_M * scale)

    left = (box_left + box_right - map_w) // 2
    top = (box_top + box_bottom - map_h) // 2
    right = left + map_w
    bottom = top + map_h

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

    # Axis labels and origin (corner 1 = bottom-left)
    cv2.putText(
        image,
        f"{FLOOR_WIDTH_IN} in (X)",
        ((left + right) // 2 - 40, bottom + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        black,
        1
    )

    cv2.putText(
        image,
        f"{FLOOR_HEIGHT_IN} in (Y)",
        (12, (top + bottom) // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        black,
        1
    )

    cv2.putText(
        image,
        "(0,0)",
        (left + 16, bottom - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        gray,
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

    # Anchor positions (small squares with their ID)
    for anchor, pos in anchor_positions_m().items():

        px, py = pixel(pos)
        px = min(max(px, left + 6), right - 6)
        py = min(max(py, top + 6), bottom - 6)

        cv2.rectangle(
            image,
            (px - 5, py - 5),
            (px + 5, py + 5),
            gray,
            -1
        )

        cv2.putText(
            image,
            f"A{anchor}",
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            gray,
            1
        )

    # BLE-only estimate (hollow blue circle)
    if ble_only is not None:

        cv2.circle(
            image,
            pixel(ble_only),
            9,
            (200, 90, 0),
            2
        )

    # Show each camera's individual estimate
    palette = [
        (200, 120, 0),
        (200, 0, 130),
        (0, 160, 160),
        (90, 90, 200)
    ]

    camera_colors = {
        anchor: palette[i % len(palette)]
        for i, anchor in enumerate(CAMERAS)
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
            f"FUSED: X={to_in(position[0]):.1f} in   "
            f"Y={to_in(position[1]):.1f} in"
        )

    else:
        label = "FUSED: unavailable"

    cv2.putText(
        image,
        label,
        (20, 512),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        black,
        2
    )

    ble_label = (
        f"BLE only: X={to_in(ble_only[0]):.0f} in  "
        f"Y={to_in(ble_only[1]):.0f} in"
        if ble_only is not None
        else "BLE only: no live anchors"
    )

    cv2.putText(
        image,
        ble_label,
        (20, 533),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 90, 0),
        1
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

    # Live anchor readings, one row per anchor
    for i, anchor in enumerate(sorted(ANCHORS)):

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

        tag = "" if anchor in CAMERAS else " ble"

        cv2.putText(
            image,
            f"{anchor} {state}{tag}",
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
                f"{weights[anchor] * 100:.0f}%"
                if anchor in weights
                else "--"
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
        (12, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        gray,
        1
    )

    return image


# ============================================================
# CAMERA SOURCES (USB index or RTSP URL)
# ============================================================

def load_secrets():
    """
    camera_secrets.json:
        {
          "user": "admin", "password": "...",        # default for all
          "cameras": { "1": {"user": "...", "password": "..."} }  # optional per anchor
        }
    Returns {anchor_or_"default": {"user", "password"}} or None if missing.
    """

    try:

        with open(SECRETS_FILE, encoding="utf-8") as f:
            data = json.load(f)

        secrets = {
            "default": {
                "user": str(data["user"]),
                "password": str(data["password"])
            }
        }

        for key, creds in data.get("cameras", {}).items():
            secrets[int(key)] = {
                "user": str(creds["user"]),
                "password": str(creds["password"])
            }

        return secrets

    except FileNotFoundError:
        return None

    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{SECRETS_FILE} is not valid: {exc}"
        )


# These cameras take 5-7 s to send their first keyframe
RTSP_TIMEOUT_MS = 15000       # connect / single read timeout
RTSP_FIRST_FRAME_S = 20.0     # wait for the first frame after opening
RTSP_STALE_S = 15.0           # no frame for this long -> reconnect


def open_rtsp(url):

    return cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_TIMEOUT_MS
        ]
    )


class RtspCamera:
    """
    Reads an RTSP stream in a background thread and always hands
    out the newest frame, so the main loop never falls behind the
    camera. Same read()/release()/isOpened() interface as
    cv2.VideoCapture.
    """

    def __init__(self, url):

        self.url = url
        self.cap = open_rtsp(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.running = self.cap.isOpened()
        self.thread = None

        if self.running:

            self.thread = threading.Thread(
                target=self._loop,
                daemon=True
            )
            self.thread.start()

    def _loop(self):

        last_frame = time.monotonic()

        while self.running:

            try:
                ok, frame = self.cap.read()

            except cv2.error:
                # Stream torn down under us (release or reconnect)
                if not self.running:
                    break
                ok, frame = False, None

            if ok:

                last_frame = time.monotonic()

                with self.lock:
                    self.frame = frame
                    self.ok = True

                continue

            if time.monotonic() - last_frame > RTSP_STALE_S:

                # Stream dropped: reconnect
                print("RTSP stream lost, reconnecting...")

                with self.lock:
                    self.ok = False

                self.cap.release()
                time.sleep(2.0)
                self.cap = open_rtsp(self.url)
                last_frame = time.monotonic()

            else:
                time.sleep(0.01)

    def isOpened(self):
        return self.running

    def read(self):

        # Wait for the first frame after opening; afterwards the
        # newest frame is returned immediately
        deadline = time.monotonic() + RTSP_FIRST_FRAME_S

        while time.monotonic() < deadline:

            with self.lock:

                if self.ok and self.frame is not None:
                    return True, self.frame.copy()

            time.sleep(0.01)

        return False, None

    def release(self):

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        self.cap.release()


def redact(url):

    return url.format(user="****", password="****")


def open_camera(anchor, source, secrets):

    if isinstance(source, int):

        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open USB camera index {source}"
            )

        print(f"Camera {anchor}: USB index {source}")
        return cap

    if "{user}" in source or "{password}" in source:

        if secrets is None:
            raise RuntimeError(
                f"Camera {anchor} needs a password but "
                f"{SECRETS_FILE} was not found"
            )

        creds = secrets.get(anchor, secrets["default"])
        url = source.format(**creds)

    else:
        url = source

    print(f"Camera {anchor}: connecting to {redact(source)}")

    cap = RtspCamera(url)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open RTSP camera {anchor} at "
            f"{redact(source)} (check IP, port, path, "
            f"user/password, and that the laptop can ping it)"
        )

    ok, frame = cap.read()

    if not ok:
        cap.release()
        raise RuntimeError(
            f"RTSP camera {anchor} connected but sent no frames"
        )

    h, w = frame.shape[:2]
    print(f"Camera {anchor}: RTSP stream {w}x{h}")

    return cap


# ============================================================
# MAIN
# ============================================================

def main():

    missing = sorted(set(CAMERAS) - set(ANCHORS))

    if missing:
        raise RuntimeError(
            f"Camera anchor(s) {missing} are not in ANCHORS; "
            f"add their floor positions"
        )

    print("BLE anchors:", {
        a: f"({x}, {y}) in" + ("" if a in CAMERAS else " standalone")
        for a, (x, y) in sorted(ANCHORS.items())
    })

    # Start BLE receiver in background thread
    threading.Thread(
        target=receive_ble,
        daemon=True
    ).start()

    caps = {}

    try:

        secrets = load_secrets()

        # Open the cameras (USB or RTSP). A camera that is not
        # configured yet or cannot be reached is skipped so the
        # rest can still run.
        for anchor, source in CAMERAS.items():

            if isinstance(source, str) and "XX" in source:
                print(
                    f"Camera {anchor}: no address set yet, skipping"
                )
                continue

            try:
                caps[anchor] = open_camera(
                    anchor,
                    source,
                    secrets
                )

            except RuntimeError as exc:
                print(f"Camera {anchor}: SKIPPED - {exc}")

        if not caps:
            raise RuntimeError("No camera could be opened")

        print(f"Running with camera(s): {sorted(caps)}")

        # Load or create calibration
        matrices, methods = load_calibration()

        for anchor, cap in caps.items():

            if anchor not in matrices:

                matrices[anchor], methods[anchor] = calibrate(
                    anchor,
                    cap
                )

        save_calibration(matrices, methods)

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
            fused, weights, status, ble_only = fuse(
                positions,
                snapshot
            )

            # Display live dashboard
            dashboard = make_dashboard(
                fused,
                positions,
                weights,
                snapshot,
                status,
                ble_only
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

                    matrices[anchor], methods[anchor] = calibrate(
                        anchor,
                        cap
                    )

                save_calibration(matrices, methods)

    finally:

        for cap in caps.values():
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
