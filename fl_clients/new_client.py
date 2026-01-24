import argparse
import pickle
import socket
import struct
import time
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--client_id", type=int, required=True)
    p.add_argument("--num_clients", type=int, default=15)
    p.add_argument("--server", type=str, default="127.0.0.1:5001", help="IP:PORT")
    return p.parse_args()


def send_msg(sock: socket.socket, payload: bytes) -> None:
    header = struct.pack("!I", len(payload))
    sock.sendall(header + payload)


def local_train(client_id: int):
    # Dummy "model": 10 numbers
    weights = np.random.rand(10)
    print(f"[Client {client_id}] Local training complete.")
    return weights


def main():
    args = parse_args()
    host, port_str = args.server.split(":")
    port = int(port_str)

    # Member B style prints (even though this is the simple stack)
    print(f"[Client] client_id={args.client_id} num_clients={args.num_clients}")
    print("[Client] local_train_examples=UNKNOWN (dummy client)")
    print("[Client] device=cpu (dummy client)")
    print(f"[Client] server={args.server}")

    while True:
        w = local_train(args.client_id)
        payload = pickle.dumps(w, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((host, port))
            send_msg(s, payload)
            s.close()
            print(f"[Client {args.client_id}] Update sent to server.")
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"[Client {args.client_id}] Server unavailable: {type(e).__name__}: {e}")
            return 1

        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
