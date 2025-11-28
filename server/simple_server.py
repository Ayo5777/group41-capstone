import socket
import pickle
import numpy as np
from datetime import datetime

HOST = "0.0.0.0"
PORT = 5001

AGG_WEIGHTS = []

def aggregate(weights_list):
    return np.mean(weights_list, axis=0)

if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(5)

    print("[Server] Waiting for client updates...")

    while True:
        conn, addr = s.accept()
        data = conn.recv(4096)
        weights = pickle.loads(data)
        AGG_WEIGHTS.append(weights)

        agg = aggregate(AGG_WEIGHTS)
        print(f"[Server] Received update from {addr}, Aggregated Weights: {agg[:3]}…")

        # Log to file
        with open("results/log.txt", "a") as f:
            f.write(f"{datetime.now()} - Received from {addr}\n")

        conn.close()
