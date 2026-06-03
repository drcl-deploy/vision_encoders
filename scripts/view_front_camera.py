#!/usr/bin/env python3
"""View the D435i camera stream from the Orin over TCP.

Usage (on desktop):
    python3 view_front_camera.py [orin_ip]

Requires camera_streamer.py running on the Orin.
"""

import cv2
import numpy as np
import socket
import struct
import sys

ORIN_IP = "192.168.123.164"
PORT = 5555


def main():
    ip = sys.argv[1] if len(sys.argv) > 1 else ORIN_IP

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, PORT))
    print(f"Connected to {ip}:{PORT}", flush=True)

    buf = b""
    while True:
        while len(buf) < 4:
            chunk = sock.recv(4096)
            if not chunk:
                print("Connection closed")
                return
            buf += chunk
        length = struct.unpack("<I", buf[:4])[0]
        buf = buf[4:]

        while len(buf) < length:
            chunk = sock.recv(65536)
            if not chunk:
                print("Connection closed")
                return
            buf += chunk
        jpg_data = buf[:length]
        buf = buf[length:]

        img = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow("Go2 Front Camera", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    sock.close()


if __name__ == "__main__":
    main()
