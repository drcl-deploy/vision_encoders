"""The one python-side implementation of the camera TCP wire format.

<uint32 LE payload length><JPEG-or-PNG bytes> per frame; cv2.imdecode sniffs
the magic bytes. Producers: camera_streamer.py (real), unitree_mujoco's
head-camera server (sim). Consumers: viewer.py, tests/fps.py, encoder_node
(C++ twin of this file).
"""

import socket
import struct

import cv2
import numpy as np


def recv_exact(sock, n):
    """n bytes or None on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock):
    """One decoded BGR frame, or None on EOF/decode failure."""
    hdr = recv_exact(sock, 4)
    if hdr is None:
        return None
    (length,) = struct.unpack('<I', hdr)
    payload = recv_exact(sock, length)
    if payload is None:
        return None
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)


def send_frame(conn, img_bgr, codec='.jpg', params=None):
    """Encode + send one frame (server side — the streamer/bench half)."""
    ok, buf = cv2.imencode(codec, img_bgr, params or [])
    if not ok:
        raise ValueError('imencode failed')
    conn.sendall(struct.pack('<I', len(buf)) + buf.tobytes())


def connect(ip, port, timeout=5.0):
    return socket.create_connection((ip, port), timeout=timeout)
