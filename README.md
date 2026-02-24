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
## 📊 Experimental Results

The framework was evaluated against 5 unseen, uniquely parameterized simulated devices to test Sim-to-Real robustness. 

**Sim-to-Real Success Rate: 100%**
The agent successfully adapted to varying lever arms and cross-talk matrices on the fly, executing non-linear, multi-axis voltage corrections to locate the target (1,0) state.

![Domain Randomization Results](results/Test%201%202%203%204%205.png)
*Evaluation across 5 randomized device profiles, demonstrating consistent target isolation.*

### Navigation and Perception
![Global Navigation Strategy](results/global%20Navigation%201.png)
*Left: Global trajectory through the charge stability diagram. Right: The final 16x16 local 2D scan representing the agent's spatial vision.*

### 3-Dot Array Scaling with Virtual Gates
![Virtual Gate Control](results/Agent%20controlling%20virtual%20%20gates.png)
*By inverting the automatically discovered cross-talk matrix, the agent successfully isolates the (1,1,1) regime. Notice how physical gates automatically compensate for induced cross-talk.*
