#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE=$1

case $PROFILE in
    wifi_good)
        bash "$SCRIPT_DIR/set_impairments.sh" 20 5 0 50mbit
        ;;
    wifi_bad)
        bash "$SCRIPT_DIR/set_impairments.sh" 150 30 5 2mbit
        ;;
    3g)
        bash "$SCRIPT_DIR/set_impairments.sh" 300 100 2 1mbit
        ;;
    congested)
        bash "$SCRIPT_DIR/set_impairments.sh" 80 40 10 500kbit
        ;;
    baseline)
        sudo tc qdisc del dev wlo1 root 2>/dev/null
        echo "[+] Baseline restored!"
        ;;
    *)
        echo "Available profiles: wifi_good, wifi_bad, 3g, congested, baseline"
        ;;
esac
