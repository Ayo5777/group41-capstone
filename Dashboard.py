import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


METRICS_DIR = Path("metrics")
LIVE_METRICS_FILE = METRICS_DIR / "live_metrics.json"
EVENTS_FILE = METRICS_DIR / "events.jsonl"

REFRESH_MS = 1500  # dashboard refresh interval


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
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except Exception:
                        continue
    except Exception:
        return []
    return lines[-max_lines:]


def client_color(status: str) -> str:
    status = (status or "").lower()
    if status == "training":
        return "#1f77b4"
    if status == "uploading":
        return "#2ca02c"
    if status == "delayed":
        return "#ff7f0e"
    if status == "dropped":
        return "#d62728"
    if status == "idle":
        return "#7f7f7f"
    if status == "complete":
        return "#17becf"
    return "#9467bd"


def server_color(status: str) -> str:
    status = (status or "").lower()
    if status == "aggregating":
        return "#2ca02c"
    if status == "waiting":
        return "#ff7f0e"
    if status == "idle":
        return "#7f7f7f"
    return "#1f77b4"


def build_metrics_frames(data: Dict[str, Any]):
    history = data.get("history", [])

    if not history:
        return pd.DataFrame(columns=["round", "global_accuracy", "global_loss", "round_time_s"])

    df = pd.DataFrame(history)
    for col in ["round", "global_accuracy", "global_loss", "round_time_s"]:
        if col not in df.columns:
            df[col] = None
    return df


def draw_topology(clients: List[Dict[str, Any]], server_status: str):
    fig = go.Figure()

    # Server at center
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

    # Place clients in a circle
    n = max(len(clients), 1)
    radius = 2.5

    xs = []
    ys = []
    labels = []
    colors = []
    hover_texts = []

    import math
    for i, c in enumerate(clients):
        angle = 2 * math.pi * i / n
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        xs.append(x)
        ys.append(y)
        cid = c.get("client_id", i)
        status = c.get("status", "unknown")
        progress = c.get("progress", 0)
        latency_ms = c.get("latency_ms", "N/A")
        packet_loss = c.get("packet_loss_pct", "N/A")

        labels.append(f"C{cid}")
        colors.append(client_color(status))
        hover_texts.append(
            f"Client {cid}<br>Status: {status}"
            f"<br>Progress: {progress}%"
            f"<br>Latency: {latency_ms} ms"
            f"<br>Packet loss: {packet_loss}%"
        )

        # edge from server to client
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


def line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
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


def progress_label(progress: int) -> str:
    return f"{int(progress)}%"


# =
st.set_page_config(
    page_title="FL Live Dashboard",
    layout="wide",
)

st.title("Federated Learning Live Dashboard")

# Auto refresh
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

data = safe_read_json(LIVE_METRICS_FILE)
events = safe_read_events(EVENTS_FILE, max_lines=30)

# Fallback example if file not present
if not data:
    st.warning("No live metrics file found yet. Waiting for metrics/live_metrics.json ...")
    st.code(
        """Expected file: metrics/live_metrics.json
Expected events log: metrics/events.jsonl"""
    )
    st.stop()


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

# =========================
# Top metrics
# =========================
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

# =========================
left, right = st.columns([1.4, 1.0])

with left:
    st.plotly_chart(draw_topology(clients, server_status), use_container_width=True)

with right:
    st.subheader("Recent Events")

    if not events:
        st.info("No events yet.")
    else:
        for ev in reversed(events[-12:]):
            ts = ev.get("timestamp", "")
            level = ev.get("level", "INFO")
            msg = ev.get("message", "")
            st.markdown(f"**[{ts}] {level}** — {msg}")

st.divider()

# =========================
# Charts
# =========================
df = build_metrics_frames(data)

c1, c2, c3 = st.columns(3)
with c1:
    if not df.empty:
        st.plotly_chart(
            line_chart(df, "round", "global_accuracy", "Accuracy vs Round", "Accuracy"),
            use_container_width=True,
        )
    else:
        st.info("No accuracy history yet.")

with c2:
    if not df.empty:
        st.plotly_chart(
            line_chart(df, "round", "global_loss", "Loss vs Round", "Loss"),
            use_container_width=True,
        )
    else:
        st.info("No loss history yet.")

with c3:
    if not df.empty:
        st.plotly_chart(
            line_chart(df, "round", "round_time_s", "Round Time vs Round", "Seconds"),
            use_container_width=True,
        )
    else:
        st.info("No round-time history yet.")

st.divider()

# =========================
# Client table/cards
# =========================
st.subheader("Client Status")

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
            st.progress(progress / 100.0, text=progress_label(progress))

st.divider()

# =========================
with st.expander("Show raw live JSON"):
    st.json(data)