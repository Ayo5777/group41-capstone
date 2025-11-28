#!/bin/bash

IFACE=wlo1   # change to your interface, e.g. eth0, enp3s0

LATENCY=$1
JITTER=$2
LOSS=$3
BANDWIDTH=$4

echo "[*] Applying network impairments…"
echo "    Latency:   ${LATENCY}ms"
echo "    Jitter:    ${JITTER}ms"
echo "    Loss:      ${LOSS}%"
echo "    Bandwidth: ${BANDWIDTH}"

sudo tc qdisc del dev $IFACE root 2>/dev/null

sudo tc qdisc add dev $IFACE root handle 1: htb default 11
sudo tc class add dev $IFACE parent 1: classid 1:1 htb rate ${BANDWIDTH}
sudo tc class add dev $IFACE parent 1:1 classid 1:11 htb rate ${BANDWIDTH}

sudo tc qdisc add dev $IFACE parent 1:11 netem delay ${LATENCY}ms ${JITTER}ms loss ${LOSS}%

echo "[+] Impairments applied!"
