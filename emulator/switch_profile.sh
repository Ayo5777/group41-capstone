#!/bin/bash

PROFILE=$1

case $PROFILE in
    wifi_good)
        bash set_impairments.sh 20 5 0 50mbit
        ;;
    wifi_bad)
        bash set_impairments.sh 150 30 5 2mbit
        ;;
    3g)
        bash set_impairments.sh 300 100 2 1mbit
        ;;
    congested)
        bash set_impairments.sh 80 40 10 500kbit
        ;;
    baseline)
        sudo tc qdisc del dev wlo1 root 2>/dev/null
        echo "[+] Baseline restored!"
        ;;
    *)
        echo "Available profiles: wifi_good, wifi_bad, 3g, congested, baseline"
        ;;
esac
