"""
Generate the four ArUco markers used for floor calibration.

    python generate_markers.py

Writes:
    marker_0.png .. marker_3.png   one marker per page, print each
                                   FULL PAGE (8-10 inch marker) for
                                   the real floor
    markers_sheet.png              all four on one letter page, for
                                   on-screen tests or small setups

Each marker's CENTER is the floor corner. Measure the 52 x 123 in
floor between marker centers. Tape the paper flat, keep the white
border clear, use matte paper if you can (gloss reflects the lights).
"""

import sys

from ble_camera_fusion import (
    ARUCO_CORNER_IDS,
    ARUCO_DICT,
    corner_labels
)

import cv2
import numpy as np

MARKER_PX = 1500     # marker side
QUIET_PX = 300       # white border all round (required for detection)
LABEL_PX = 220       # text strip under the marker

SHEET_W, SHEET_H = 2550, 3300    # US letter at 300 dpi


def make_page(dictionary, marker_id, corner, label):

    marker = cv2.aruco.generateImageMarker(
        dictionary,
        marker_id,
        MARKER_PX
    )

    page_w = MARKER_PX + 2 * QUIET_PX
    page_h = MARKER_PX + 2 * QUIET_PX + LABEL_PX

    page = np.full((page_h, page_w), 255, dtype=np.uint8)

    page[
        QUIET_PX:QUIET_PX + MARKER_PX,
        QUIET_PX:QUIET_PX + MARKER_PX
    ] = marker

    # Tick marks on the page edges aligned with the marker center,
    # for lining up a tape measure
    cx = QUIET_PX + MARKER_PX // 2
    cy = QUIET_PX + MARKER_PX // 2
    tick = QUIET_PX // 3

    cv2.line(page, (cx, 0), (cx, tick), 0, 6)
    cv2.line(page, (cx, page_h - LABEL_PX - tick), (cx, page_h - LABEL_PX), 0, 6)
    cv2.line(page, (0, cy), (tick, cy), 0, 6)
    cv2.line(page, (page_w - tick, cy), (page_w, cy), 0, 6)

    cv2.putText(
        page,
        f"ID {marker_id} = corner {corner + 1}  {label} in",
        (QUIET_PX, page_h - LABEL_PX + 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        0,
        6
    )

    cv2.putText(
        page,
        "marker center = floor corner",
        (QUIET_PX, page_h - LABEL_PX + 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        0,
        4
    )

    return page


def main():

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    labels = corner_labels()

    pages = []

    for marker_id, corner in sorted(
        ARUCO_CORNER_IDS.items(),
        key=lambda kv: kv[1]
    ):

        page = make_page(
            dictionary,
            marker_id,
            corner,
            labels[corner]
        )

        filename = f"marker_{marker_id}.png"
        cv2.imwrite(filename, page)
        pages.append((filename, marker_id, page))

        print(f"wrote {filename}  (corner {corner + 1} {labels[corner]} in)")

    # 2x2 sheet
    sheet = np.full((SHEET_H, SHEET_W), 255, dtype=np.uint8)
    cell_w, cell_h = SHEET_W // 2, SHEET_H // 2

    for i, (_, _, page) in enumerate(pages):

        scale = min(
            (cell_w - 100) / page.shape[1],
            (cell_h - 100) / page.shape[0]
        )

        small = cv2.resize(
            page,
            (int(page.shape[1] * scale), int(page.shape[0] * scale)),
            interpolation=cv2.INTER_AREA
        )

        x = (i % 2) * cell_w + (cell_w - small.shape[1]) // 2
        y = (i // 2) * cell_h + (cell_h - small.shape[0]) // 2

        sheet[y:y + small.shape[0], x:x + small.shape[1]] = small

    cv2.imwrite("markers_sheet.png", sheet)
    print("wrote markers_sheet.png  (all four, for screen tests)")

    # Self-check: every file must decode to its own id
    detector = cv2.aruco.ArucoDetector(
        dictionary,
        cv2.aruco.DetectorParameters()
    )

    failed = False

    for filename, marker_id, page in pages:

        _, ids, _ = detector.detectMarkers(page)
        got = [] if ids is None else ids.ravel().tolist()

        if got == [marker_id]:
            print(f"check {filename} -> id {marker_id} OK")
        else:
            print(f"check {filename} -> FAILED, detected {got}")
            failed = True

    _, ids, _ = detector.detectMarkers(sheet)
    got = sorted([] if ids is None else ids.ravel().tolist())
    expected = sorted(ARUCO_CORNER_IDS)

    if got == expected:
        print(f"check markers_sheet.png -> ids {got} OK")
    else:
        print(f"check markers_sheet.png -> FAILED, detected {got}")
        failed = True

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
