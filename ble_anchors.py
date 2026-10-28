
import socket
import threading
import time

# ==========================================
# WIFI BLE ANCHOR RECEIVER
# ==========================================

# All anchors send data to this UDP port
UDP_IP = "0.0.0.0"
UDP_PORT = 5005

# Anchors to monitor
ANCHOR_IDS = [1, 3]

# Timeout before anchor becomes inactive
TIMEOUT = 3.0

anchors = {}
lock = threading.Lock()


# ==========================================
# RECEIVE DATA FROM ESP32 ANCHORS
# ==========================================

def receive_anchors():

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)

    print(f"Listening for ESP32 anchors on UDP port {UDP_PORT}")

    while True:

        try:
            data, address = sock.recvfrom(1024)

            line = data.decode(
                "utf-8", errors="ignore"
            ).strip()

            if not line.startswith("DATA,"):
                continue

            parts = line.split(",")

            anchor_id = int(parts[1])

            if anchor_id not in ANCHOR_IDS:
                continue

            # Phone lost
            if len(parts) == 3 and parts[2] == "LOST":

                with lock:
                    anchors.pop(anchor_id, None)

                print(f"Anchor {anchor_id}: Phone lost")
                continue

            if len(parts) != 5:
                continue

            rssi = float(parts[2])
            distance = float(parts[3])
            score = float(parts[4])

            with lock:
                anchors[anchor_id] = {
                    "rssi": rssi,
                    "distance": distance,
                    "score": score,
                    "time": time.monotonic(),
                    "ip": address[0]
                }

        except socket.timeout:
            continue

        except (ValueError, IndexError):
            continue

        except OSError as e:
            print(f"UDP socket error: {e}")
            break

    sock.close()


# ==========================================
# START RECEIVER THREAD
# ==========================================

thread = threading.Thread(
    target=receive_anchors,
    daemon=True
)

thread.start()


# ==========================================
# DISPLAY ANCHOR DATA
# ==========================================

print("Waiting for BLE anchors over WiFi...")

try:
    while True:

        with lock:
            now = time.monotonic()

            # Remove stale anchor measurements
            stale = [
                anchor_id
                for anchor_id, data in anchors.items()
                if now - data["time"] >= TIMEOUT
            ]

            for anchor_id in stale:
                del anchors[anchor_id]

            active = {
                anchor_id: data.copy()
                for anchor_id, data in anchors.items()
            }

        if active:

            # Higher RSSI = stronger signal
            closest = max(
                active,
                key=lambda a: active[a]["rssi"]
            )

            print("\n-----------------------------")

            for anchor_id in ANCHOR_IDS:

                if anchor_id in active:

                    data = active[anchor_id]

                    print(
                        f"Anchor {anchor_id}: "
                        f"{data['rssi']:.1f} dBm | "
                        f"{data['distance']:.2f} m"
                    )

                else:
                    print(
                        f"Anchor {anchor_id}: No signal"
                    )

            print(f"\nCLOSEST ANCHOR: {closest}")

        else:
            print("Waiting for phone...")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopping WiFi BLE receiver...")
