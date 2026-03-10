import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


METRICS_DIR = Path("metrics")
LIVE_METRICS_FILE = METRICS_DIR / "live_metrics.json"
EVENTS_FILE = METRICS_DIR / "events.jsonl"


DEFAULT_STATE = {
    "server": {
        "status": "idle"
    },
    "network": {
        "latency_ms": 0,
        "packet_loss_pct": 0,
        "bandwidth_mbps": 0
    },
    "current": {
        "round": 0,
        "global_accuracy": 0.0,
        "global_loss": 0.0,
        "round_time_s": 0.0,
        "updates_received": 0,
        "updates_expected": 0
    },
    "clients": [],
    "history": []
}


def _ensure_dir():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _read_state() -> Dict[str, Any]:
    _ensure_dir()
    if not LIVE_METRICS_FILE.exists():
        return deepcopy(DEFAULT_STATE)
    try:
        with open(LIVE_METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return deepcopy(DEFAULT_STATE)


def _write_state(state: Dict[str, Any]) -> None:
    _ensure_dir()
    with open(LIVE_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def log_event(level: str, message: str) -> None:
    _ensure_dir()
    event = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "level": level.upper(),
        "message": message,
    }
    with open(EVENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def initialize_dashboard(
    num_clients: int,
    latency_ms: int = 0,
    packet_loss_pct: float = 0,
    bandwidth_mbps: float = 0
) -> None:
    state = deepcopy(DEFAULT_STATE)
    state["network"] = {
        "latency_ms": latency_ms,
        "packet_loss_pct": packet_loss_pct,
        "bandwidth_mbps": bandwidth_mbps,
    }
    state["current"]["updates_expected"] = num_clients
    state["clients"] = [
        {
            "client_id": i,
            "status": "idle",
            "progress": 0,
            "local_accuracy": None,
            "local_loss": None,
            "latency_ms": latency_ms,
            "packet_loss_pct": packet_loss_pct,
        }
        for i in range(num_clients)
    ]
    _write_state(state)
    log_event("INFO", f"Dashboard initialized for {num_clients} clients")


def update_network(
    latency_ms: Optional[int] = None,
    packet_loss_pct: Optional[float] = None,
    bandwidth_mbps: Optional[float] = None,
) -> None:
    state = _read_state()

    if latency_ms is not None:
        state["network"]["latency_ms"] = latency_ms
    if packet_loss_pct is not None:
        state["network"]["packet_loss_pct"] = packet_loss_pct
    if bandwidth_mbps is not None:
        state["network"]["bandwidth_mbps"] = bandwidth_mbps

    for c in state["clients"]:
        if latency_ms is not None:
            c["latency_ms"] = latency_ms
        if packet_loss_pct is not None:
            c["packet_loss_pct"] = packet_loss_pct

    _write_state(state)


def update_server_status(status: str) -> None:
    state = _read_state()
    state["server"]["status"] = status
    _write_state(state)


def start_round(round_num: int, updates_expected: Optional[int] = None) -> None:
    state = _read_state()
    state["current"]["round"] = round_num
    state["current"]["updates_received"] = 0
    if updates_expected is not None:
        state["current"]["updates_expected"] = updates_expected
    state["server"]["status"] = "waiting"

    for c in state["clients"]:
        c["status"] = "idle"
        c["progress"] = 0
        c["local_accuracy"] = None
        c["local_loss"] = None

    _write_state(state)
    log_event("INFO", f"Round {round_num} started")


def update_client(
    client_id: int,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    local_accuracy: Optional[float] = None,
    local_loss: Optional[float] = None,
    latency_ms: Optional[int] = None,
    packet_loss_pct: Optional[float] = None,
) -> None:
    state = _read_state()

    for c in state["clients"]:
        if c["client_id"] == client_id:
            if status is not None:
                c["status"] = status
            if progress is not None:
                c["progress"] = int(progress)
            if local_accuracy is not None:
                c["local_accuracy"] = float(local_accuracy)
            if local_loss is not None:
                c["local_loss"] = float(local_loss)
            if latency_ms is not None:
                c["latency_ms"] = latency_ms
            if packet_loss_pct is not None:
                c["packet_loss_pct"] = packet_loss_pct
            break

    _write_state(state)


def mark_client_uploaded(client_id: int) -> None:
    state = _read_state()

    for c in state["clients"]:
        if c["client_id"] == client_id:
            c["status"] = "uploading"
            c["progress"] = 100
            break

    state["current"]["updates_received"] += 1
    _write_state(state)
    log_event(
        "INFO",
        f"Client {client_id} uploaded update "
        f"({state['current']['updates_received']}/{state['current']['updates_expected']})"
    )


def finish_round(
    round_num: int,
    global_accuracy: float,
    global_loss: float,
    round_time_s: float
) -> None:
    state = _read_state()
    state["server"]["status"] = "aggregating"
    _write_state(state)

    state = _read_state()
    state["current"]["round"] = round_num
    state["current"]["global_accuracy"] = float(global_accuracy)
    state["current"]["global_loss"] = float(global_loss)
    state["current"]["round_time_s"] = float(round_time_s)
    state["server"]["status"] = "idle"

    state["history"].append({
        "round": round_num,
        "global_accuracy": float(global_accuracy),
        "global_loss": float(global_loss),
        "round_time_s": float(round_time_s),
    })

    _write_state(state)
    log_event("INFO", f"Aggregation complete for round {round_num}")