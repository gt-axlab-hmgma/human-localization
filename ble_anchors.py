
import serial
import threading
import time

PORTS = {
    1: "COM4",
    2: "COM3",
    4: "COM6",
}

BAUD_RATE = 115200
TIMEOUT = 3.0

anchors = {}
lock = threading.Lock()


def read_anchor(anchor_id, port):

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)

        print(f"Anchor {anchor_id} connected on {port}")

        while True:
            line = ser.readline().decode(
                "utf-8", errors="ignore"
            ).strip()
            #if line.startswith(("DATA,", "LOST,")):
                #print(f"[COM {port}] {line}")

            if line.startswith("DATA,"):
                try:
                    parts = line.split(",")

                    reported_id = int(parts[1])
                    rssi = float(parts[2])
                    distance = float(parts[3])
                    score = float(parts[4])

                    if reported_id != anchor_id:
                        continue

                    with lock:
                        anchors[anchor_id] = {
                            "rssi": rssi,
                            "distance": distance,
                            "score": score,
                            "time": time.monotonic()
                        }

                except (ValueError, IndexError):
                    continue

            elif line.startswith("LOST,"):
                with lock:
                    anchors.pop(anchor_id, None)

    except serial.SerialException as e:
        print(f"Anchor {anchor_id} error: {e}")


for anchor_id, port in PORTS.items():

    thread = threading.Thread(
        target=read_anchor,
        args=(anchor_id, port),
        daemon=True
    )

    thread.start()


while True:

    with lock:
        now = time.monotonic()

        active = {
            anchor_id: data.copy()
            for anchor_id, data in anchors.items()
            if now - data["time"] < TIMEOUT
        }

    if active:

        # Higher RSSI (less negative) = stronger signal.
        closest = max(
            active,
            key=lambda a: active[a]["rssi"]
        )

        print("\n-----------------------------")

        for anchor_id in sorted(PORTS):

            if anchor_id in active:
                data = active[anchor_id]

                print(
                    f"Anchor {anchor_id}: "
                    f"{data['rssi']:.1f} dBm | "
                    f"{data['distance']:.2f} m"
                )
            else:
                print(f"Anchor {anchor_id}: No signal")

        print(f"\nCLOSEST ANCHOR: {closest}")

    else:
        print("Waiting for phone...")

    time.sleep(0.5)
