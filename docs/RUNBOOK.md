# Runbook (Baseline)

## Prereqs
- Python 3.10+
- Install deps:
  - python -m venv .venv
  - .\.venv\Scripts\Activate.ps1
  - pip install -r requirements.txt

## Baseline (single machine quick check)
- Start server:
  - scripts\run_server.ps1
- Start clients (default 15):
  - scripts\run_clients.ps1 -Server "127.0.0.1:8080" -NumClients 15

## Two-laptop baseline
- Laptop A runs server:
  - scripts\run_server.ps1 -Host "0.0.0.0" -Port 8080
- Laptop B runs clients:
  - scripts\run_clients.ps1 -Server "192.168.0.10:8080" -NumClients 15

## Logs
- logs\client_*.log for each client process
- server logs depend on server implementation (CSV/logs folder if enabled)
