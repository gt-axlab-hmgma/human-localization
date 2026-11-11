"""
Find RTSP cameras on the local network.

    python find_cameras.py            # scans the laptop's /24 subnet
    python find_cameras.py 10.0.0     # scan a specific /24 prefix

Lists every host answering RTSP on port 10554 (this camera model)
or 554 (most others), with the server name it reports.
"""

import concurrent.futures as cf
import socket
import sys

PORTS = (10554, 554)


def local_prefix():

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip.rsplit(".", 1)[0]


def probe(target):

    ip, port = target

    try:
        s = socket.create_connection((ip, port), timeout=0.6)
        s.sendall(
            f"OPTIONS rtsp://{ip}:{port}/ RTSP/1.0\r\n"
            f"CSeq: 1\r\n\r\n".encode()
        )
        s.settimeout(1.5)
        reply = s.recv(400).decode(errors="replace")
        s.close()

    except OSError:
        return None

    server = [
        line for line in reply.split("\r\n")
        if line.lower().startswith("server")
    ]

    return ip, port, (
        server[0] if server else reply.split("\r\n")[0]
    )


def main():

    prefix = sys.argv[1] if len(sys.argv) > 1 else local_prefix()

    print(f"Scanning {prefix}.1-254 on ports {PORTS} ...")

    targets = [
        (f"{prefix}.{i}", port)
        for i in range(1, 255)
        for port in PORTS
    ]

    found = 0

    with cf.ThreadPoolExecutor(64) as ex:

        for result in ex.map(probe, targets):

            if result:
                ip, port, server = result
                found += 1
                print(
                    f"  {ip}:{port}  ->  {server}\n"
                    f"      rtsp://{{user}}:{{password}}@{ip}:{port}/tcp/av0_0"
                )

    print(f"{found} camera(s) found" if found else "No cameras found")


if __name__ == "__main__":
    main()
