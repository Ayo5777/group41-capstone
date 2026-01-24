import os
import socket
import pickle
import struct
import numpy as np
from datetime import datetime

HOST = "0.0.0.0"
PORT = 5001

RESULTS_DIR = "results"
LOG_PATH = os.path.join(RESULTS_DIR, "log.txt")

AGG_WEIGHTS = []


def recv_exact(conn: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes or raise ConnectionError."""
    chunks = []
    bytes_recd = 0
    while bytes_recd < n:
        chunk = conn.recv(n - bytes_recd)
        if not chunk:
            raise ConnectionError("Socket closed before receiving full message")
        chunks.append(chunk)
        bytes_recd += len(chunk)
    return b"".join(chunks)


def recv_msg(conn: socket.socket) -> bytes:
    """Receive a length-prefixed message."""
    raw_len = recv_exact(conn, 4)
    msg_len = struct.unpack("!I", raw_len)[0]
    return recv_exact(conn, msg_len)


def aggregate(weights_list):
    return np.mean(weights_list, axis=0)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(32)

    print(f"[Server] Listening on {HOST}:{PORT}")
    print(f"[Server] Logging to {LOG_PATH}")

    while True:
        conn, addr = s.accept()
        try:
            data = recv_msg(conn)
            weights = pickle.loads(data)
            AGG_WEIGHTS.append(weights)

            agg = aggregate(AGG_WEIGHTS)
            print(f"[Server] Received update from {addr}, Aggregated Weights: {agg[:3]}…")

            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} - Received from {addr}\n")
        except Exception as e:
            print(f"[Server] Error handling client {addr}: {type(e).__name__}: {e}")
        finally:
            conn.close()
