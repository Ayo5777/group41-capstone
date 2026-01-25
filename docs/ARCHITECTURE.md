# Architecture (WIP)

## Physical setup (baseline)
- Laptop A: FL Server
- Laptop B: Multiple FL clients (target: 15)
- Baseline link: direct Ethernet
- Later: NE-ONE inserted inline (bridge mode) between the two laptops

## Repo modules
- server/: FL server code (Flower, FedAvg, logging)
- fl_clients/: FL client code (client_id, dataset shard mapping)
- emulator/: network impairment tooling (tc/netem scripts now; NE-ONE profiles later)
- scripts/: orchestration scripts (start/stop clients/server)

## Notes
Implementation is staged:
1) baseline FL runs reliably
2) scale to N clients
3) add impairment profiles (tc, then NE-ONE)
