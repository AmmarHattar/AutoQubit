# AutoQubit: Autonomous Spin Qubit Calibration

A fully autonomous, closed-loop artificial intelligence framework designed to calibrate semiconductor spin qubits in silicon overlapping-gate quantum dot (Si-OG-QD) devices. 

## 🧠 Core Architecture
1. **Perception (CNN):** Processes noisy, local 16x16 2D charge sensor scans to extract latent spatial features.
2. **Control (SAC):** A Soft Actor-Critic agent fuses the CNN's latent vector with global voltage coordinates to output continuous gate adjustments.

## ✨ Key Features
* **Active Sensing:** The agent controls measurement integration time, executing fast scans in empty space and slow, high-fidelity scans near transition boundaries.
* **Domain Randomization:** The simulator randomizes charging energies and lever arms every episode to ensure Sim-to-Real robustness.
* **Virtual Gates:** Automatically discovers and inverts the cross-talk matrix to navigate multi-dot arrays orthogonally.

## 🚀 Installation
pip install -r requirements.txt

## 🛠️ Usage
To train the Double Quantum Dot (DQD) agent with Domain Randomization, or the Triple Quantum Dot (TQD) agent with Virtual Gates:
python scripts/train_models.py
