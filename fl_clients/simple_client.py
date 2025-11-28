import socket
import pickle
import time
import numpy as np

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5001

def local_train():
    # Dummy "model": 10 numbers
    weights = np.random.rand(10)
    print("[Client] Local training complete.")
    return weights

def send_update(weights):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_IP, SERVER_PORT))
    data = pickle.dumps(weights)
    s.sendall(data)
    s.close()
    print("[Client] Update sent to server.")

if __name__ == "__main__":
    while True:
        w = local_train()
        send_update(w)
        time.sleep(5)  # simulate FL rounds
