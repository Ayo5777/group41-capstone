import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# CONFIGURE THESE PATHS FOR YOUR PROJECT
# =========================================================
APP_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = APP_DIR / "server" / "server.py"
CLIENT_SCRIPT = APP_DIR / "fl_clients" / "new_client.py"

METRICS_DIR = APP_DIR / "metrics"
LIVE_METRICS_FILE = METRICS_DIR / "live_metrics.json"
EVENTS_FILE = METRICS_DIR / "events.jsonl"

REFRESH_MS = 1500


# =========================================================
# SESSION STATE
# =========================================================
if "server_process" not in st.session_state:
    st.session_state.server_process = None

if "client_process" not in st.session_state:
    st.session_state.client_process = None


# =========================================================
# UTILITIES
# =========================================================
def get_python_executable() -> str:
    return sys.executable


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def process_is_running(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return

    if proc.poll() is not None:
        return

    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def launch_process(cmd: List[str], cwd: Path) -> subprocess.Popen:
    if os.name == "nt":
        return subprocess.Popen(
            cmd,
            cwd=str(cwd),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        return subprocess.Popen(
            cmd,
            cwd=str(cwd),
            preexec_fn=os.setsid,
        )


def safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def safe_read_events(path: Path, max_lines: int = 50) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows[-max_lines:]


def reset_metrics_files() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if LIVE_METRICS_FILE.exists():
        LIVE_METRICS_FILE.unlink()
    if EVENTS_FILE.exists():
        EVENTS_FILE.unlink()


def client_color(status: str) -> str:
    s = (status or "").lower()
    if s == "training":
        return "#1f77b4"
    if s == "uploading":
        return "#2ca02c"
    if s == "delayed":
        return "#ff7f0e"
    if s == "dropped":
        return "#d62728"
    if s == "complete":
        return "#17becf"
    if s == "idle":
        return "#7f7f7f"
    return "#9467bd"


def server_color(status: str) -> str:
    s = (status or "").lower()
    if s == "aggregating":
        return "#2ca02c"
    if s == "waiting":
        return "#ff7f0e"
    if s == "idle":
        return "#7f7f7f"
    return "#1f77b4"


def build_metrics_frame(data: Dict[str, Any]) -> pd.DataFrame:
    history = data.get("history", [])
    if not history:
        return pd.DataFrame(columns=["round", "global_accuracy", "global_loss", "round_time_s"])
    df = pd.DataFrame(history)
    for col in ["round", "global_accuracy", "global_loss", "round_time_s"]:
        if col not in df.columns:
            df[col] = None
    return df


def line_chart(df: pd.DataFrame, y_col: str, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["round"],
            y=df[y_col],
            mode="lines+markers",
            name=y_col,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Round",
        yaxis_title=y_label,
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def draw_topology(clients: List[Dict[str, Any]], server_status: str) -> go.Figure:
    import math

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers+text",
            marker=dict(size=40, color=server_color(server_status)),
            text=["Server"],
            textposition="top center",
            name="Server",
            hovertemplate=f"Server<br>Status: {server_status}<extra></extra>",
        )
    )

    n = max(len(clients), 1)
    radius = 2.5

    xs, ys, labels, colors, hover_texts = [], [], [], [], []

    for i, c in enumerate(clients):
        angle = 2 * math.pi * i / n
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        cid = c.get("client_id", i)
        status = c.get("status", "unknown")
        progress = c.get("progress", 0)
        latency_ms = c.get("latency_ms", "N/A")
        packet_loss = c.get("packet_loss_pct", "N/A")

        xs.append(x)
        ys.append(y)
        labels.append(f"C{cid}")
        colors.append(client_color(status))
        hover_texts.append(
            f"Client {cid}<br>Status: {status}"
            f"<br>Progress: {progress}%"
            f"<br>Latency: {latency_ms} ms"
            f"<br>Packet loss: {packet_loss}%"
        )

        fig.add_trace(
            go.Scatter(
                x=[0, x],
                y=[0, y],
                mode="lines",
                line=dict(width=2, color="rgba(180,180,180,0.6)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(size=28, color=colors),
            text=labels,
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo="text",
            name="Clients",
        )
    )

    fig.update_layout(
        title="Federated Learning Topology",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        height=450,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def render_dashboard() -> None:
    data = safe_read_json(LIVE_METRICS_FILE)
    events = safe_read_events(EVENTS_FILE, max_lines=30)

    st.subheader("Live Monitoring")

    if not data:
        st.info("No live metrics found yet. Start the server and clients first.")
        return

    server = data.get("server", {})
    network = data.get("network", {})
    clients = data.get("clients", [])
    current = data.get("current", {})

    server_status = server.get("status", "unknown")
    current_round = current.get("round", 0)
    global_accuracy = current.get("global_accuracy", 0.0)
    global_loss = current.get("global_loss", 0.0)
    round_time_s = current.get("round_time_s", 0.0)
    updates_received = current.get("updates_received", 0)
    updates_expected = current.get("updates_expected", len(clients))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Server Status", server_status)
    m2.metric("Round", current_round)
    m3.metric("Global Accuracy", f"{global_accuracy:.4f}")
    m4.metric("Global Loss", f"{global_loss:.4f}")
    m5.metric("Round Time (s)", f"{round_time_s:.2f}")

    m6, m7, m8, m9 = st.columns(4)
    m6.metric("Latency (ms)", network.get("latency_ms", "N/A"))
    m7.metric("Packet Loss (%)", network.get("packet_loss_pct", "N/A"))
    m8.metric("Bandwidth (Mbps)", network.get("bandwidth_mbps", "N/A"))
    m9.metric("Updates", f"{updates_received}/{updates_expected}")

    st.divider()

    left, right = st.columns([1.4, 1.0])

    with left:
        st.plotly_chart(draw_topology(clients, server_status), width="stretch")

    with right:
        st.markdown("**Recent Events**")
        if not events:
            st.info("No events yet.")
        else:
            for ev in reversed(events[-12:]):
                ts = ev.get("timestamp", "")
                level = ev.get("level", "INFO")
                msg = ev.get("message", "")
                st.markdown(f"**[{ts}] {level}** — {msg}")

    st.divider()

    df = build_metrics_frame(data)
    c1, c2, c3 = st.columns(3)

    with c1:
        if not df.empty:
            st.plotly_chart(
                line_chart(df, "global_accuracy", "Accuracy vs Round", "Accuracy"),
                width="stretch",
            )
        else:
            st.info("No accuracy history yet.")

    with c2:
        if not df.empty:
            st.plotly_chart(
                line_chart(df, "global_loss", "Loss vs Round", "Loss"),
                width="stretch",
            )
        else:
            st.info("No loss history yet.")

    with c3:
        if not df.empty:
            st.plotly_chart(
                line_chart(df, "round_time_s", "Round Time vs Round", "Seconds"),
                width="stretch",
            )
        else:
            st.info("No round-time history yet.")

    st.divider()

    st.markdown("**Client Status**")
    if not clients:
        st.info("No client data available.")
    else:
        cols = st.columns(3)
        for i, c in enumerate(clients):
            with cols[i % 3]:
                cid = c.get("client_id", i)
                status = c.get("status", "unknown")
                progress = int(c.get("progress", 0))
                train_loss = c.get("local_loss", None)
                train_acc = c.get("local_accuracy", None)
                latency_ms = c.get("latency_ms", "N/A")
                packet_loss = c.get("packet_loss_pct", "N/A")

                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #333;
                        border-radius:12px;
                        padding:14px;
                        margin-bottom:12px;
                        background-color:#111;
                    ">
                        <h4 style="margin:0 0 8px 0;">Client {cid}</h4>
                        <p style="margin:0;"><b>Status:</b> {status}</p>
                        <p style="margin:0;"><b>Latency:</b> {latency_ms} ms</p>
                        <p style="margin:0;"><b>Packet Loss:</b> {packet_loss}%</p>
                        <p style="margin:0;"><b>Local Accuracy:</b> {train_acc if train_acc is not None else 'N/A'}</p>
                        <p style="margin:0 0 8px 0;"><b>Local Loss:</b> {train_loss if train_loss is not None else 'N/A'}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(progress / 100.0, text=f"{progress}%")

    with st.expander("Show raw live JSON"):
        st.json(data)


# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="FL Control Panel", layout="wide")
st.title("Federated Learning Control Panel")

st.markdown(
    f"""
    <script>
        setTimeout(function(){{
            window.location.reload();
        }}, {REFRESH_MS});
    </script>
    """,
    unsafe_allow_html=True,
)

local_ip = get_local_ip()

with st.sidebar:
    st.header("Role")
    role = st.radio("Choose what to run", ["Server", "Client"], index=0)

    st.divider()
    st.markdown("**Helpers**")
    st.write(f"Python: `{get_python_executable()}`")
    st.write(f"Local IP: `{local_ip}`")
    st.write(f"Server script exists: `{SERVER_SCRIPT.exists()}`")
    st.write(f"Client script exists: `{CLIENT_SCRIPT.exists()}`")

    if st.button("Reset Metrics Files"):
        reset_metrics_files()
        st.success("Metrics files reset.")


# =========================================================
# SERVER CONTROLS
# =========================================================
if role == "Server":
    st.subheader("Server Launcher")

    col1, col2 = st.columns(2)

    with col1:
        test_name = st.text_input("Test Name", value="demo_run")
        num_clients = st.number_input("Number of Clients", min_value=1, value=5, step=1)
        rounds = st.number_input("Rounds", min_value=1, value=3, step=1)
        round_timeout = st.number_input("Round Timeout (s)", min_value=1.0, value=60.0, step=1.0)

    with col2:
        latency_ms = st.number_input("Latency (ms)", min_value=0, value=100, step=1)
        packet_loss_pct = st.number_input("Packet Loss (%)", min_value=0.0, value=2.0, step=0.5)
        bandwidth_mbps = st.number_input("Bandwidth (Mbps)", min_value=0.0, value=20.0, step=1.0)
        server_bind = st.text_input("Server Bind Address", value="0.0.0.0:8080")

    s1, s2, s3 = st.columns(3)

    with s1:
        if st.button("Start Server", width="stretch"):
            if process_is_running(st.session_state.server_process):
                st.warning("Server is already running.")
            elif not SERVER_SCRIPT.exists():
                st.error(f"Server script not found: {SERVER_SCRIPT}")
            else:
                cmd = [
                    get_python_executable(),
                    str(SERVER_SCRIPT),
                    "--test_name", test_name,
                    "--num_clients", str(num_clients),
                    "--rounds", str(rounds),
                    "--round_timeout", str(round_timeout),
                    "--latency_ms", str(latency_ms),
                    "--packet_loss_pct", str(packet_loss_pct),
                    "--bandwidth_mbps", str(bandwidth_mbps),
                ]
                st.session_state.server_process = launch_process(cmd, APP_DIR)
                st.success("Server started.")

    with s2:
        if st.button("Stop Server", width="stretch"):
            if process_is_running(st.session_state.server_process):
                stop_process(st.session_state.server_process)
                st.session_state.server_process = None
                st.success("Server stopped.")
            else:
                st.info("No running server process found in this dashboard session.")

    with s3:
        running = process_is_running(st.session_state.server_process)
        st.metric("Server Process", "Running" if running else "Stopped")

    st.caption(
        "Use the IP shown in the sidebar for client connections on other machines. "
        "Example: 192.168.x.x:8080"
    )


# =========================================================
# CLIENT CONTROLS
# =========================================================
if role == "Client":
    st.subheader("Client Launcher")

    col1, col2 = st.columns(2)

    with col1:
        client_id = st.number_input("Client ID", min_value=0, value=0, step=1)
        num_clients = st.number_input("Total Number of Clients", min_value=1, value=5, step=1)
        server_addr = st.text_input("Server Address", value=f"{local_ip}:8080")
        batch_size = st.number_input("Batch Size", min_value=1, value=32, step=1)

    with col2:
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        data_dir = st.text_input("Data Directory", value=str(APP_DIR / "DataPart" / "data"))
        force_cpu = st.checkbox("Force CPU", value=False)

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Start Client", width="stretch"):
            if process_is_running(st.session_state.client_process):
                st.warning("Client is already running.")
            elif not CLIENT_SCRIPT.exists():
                st.error(f"Client script not found: {CLIENT_SCRIPT}")
            else:
                cmd = [
                    get_python_executable(),
                    str(CLIENT_SCRIPT),
                    "--client_id", str(client_id),
                    "--num_clients", str(num_clients),
                    "--server", server_addr,
                    "--batch_size", str(batch_size),
                    "--seed", str(seed),
                    "--data_dir", data_dir,
                ]
                if force_cpu:
                    cmd.append("--cpu")

                st.session_state.client_process = launch_process(cmd, APP_DIR)
                st.success("Client started.")

    with c2:
        if st.button("Stop Client", width="stretch"):
            if process_is_running(st.session_state.client_process):
                stop_process(st.session_state.client_process)
                st.session_state.client_process = None
                st.success("Client stopped.")
            else:
                st.info("No running client process found in this dashboard session.")

    with c3:
        running = process_is_running(st.session_state.client_process)
        st.metric("Client Process", "Running" if running else "Stopped")


st.divider()
render_dashboard()