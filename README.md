# Group 41 – Wireless Network Emulator for Federated Learning

This repository contains the implementation for a wireless network emulator designed to test
Federated Learning (FL) models under realistic wireless conditions.

## Project Components

### 1. Emulator (Network Layer)
- Applies latency, jitter, packet loss, and bandwidth constraints
- Built using Linux Traffic Control (tc)
- Will integrate with Mininet-WiFi and NE-ONE hardware

### 2. FL Clients (Compute Layer)
- PyTorch / TensorFlow Federated scripts
- Runs multiple clients simulating federated learning rounds

### 3. Server (Aggregation Layer)
- FL model orchestration
- Logging of convergence, accuracy, communication delays

### 4. Scripts
- Helper scripts for automation, testing, profiling

### 5. Docs
- Architecture diagrams, meeting notes, design documents

### Progress (as of today)
- Podman installed (Docker-compatible)
- Traffic Control (tc) installed and working
- Repo initialized and structured
- Next steps: FL environment setup, emulator scripts, Mininet VM

## Team Members
- Ayomikun Oyefeso
- Diego Attard
- Julian Tiqui
- Andreas
