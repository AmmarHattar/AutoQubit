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

### 1. Sim-to-Real Transfer & Hardware Robustness
The framework was evaluated against 5 unseen, uniquely parameterized simulated devices to test Domain Randomization robustness. The agent successfully adapted to varying lever arms and cross-talk matrices on the fly, locating the target (1,0) state with a 100% success rate.

![Domain Randomization Results](results/Test%201%202%203%204%205.png)

### 2. Autonomous Navigation & Perception
By fusing a Convolutional Neural Network (CNN) with a Soft Actor-Critic (SAC) agent, the system navigates the high-dimensional voltage space directly from noisy, local 2D charge sensor scans.

![Global Navigation Strategy](results/global%20Navigation%201.png)
*Global trajectory mapping paired with the final 16x16 local 2D scan representing the agent's spatial vision.*

![SAC Agent Navigation](results/SAC%20Agent%201.png)
*The SAC agent safely traversing the charge stability diagram from deep depletion to the target transition boundary.*

![Autonomous Path Execution](results/Autonomous%20path%201.png)
*Detailed autonomous path execution demonstrating multi-axis voltage control.*

### 3. Scaling to a 3-Dot Array with Virtual Gates
Scaling to multiple quantum dots introduces severe next-nearest-neighbor cross-talk. By automatically discovering and inverting the capacitive cross-talk matrix, the AI commands "Virtual Gates." 

![3 Gates Simultaneous Control](results/Agent%20controlling%203%20gates%201.png)
*Initial attempts at 3-dot control highlight the complexities of navigating uncompensated physical gates.*

![Agent Controlling Gates](results/Agent%20controlling%20gates%20.png)
*The hardware auto-compensating during the tuning sequence based on the inverted matrix.*

![Virtual Gate Control Success](results/Agent%20controlling%20virtual%20%20gates.png)
*Success! Using Virtual Gates, the physical gates automatically compensate for induced cross-talk, driving the physical voltages into negative space precisely enough to lock all three dots into the (1,1,1) electron regime simultaneously.*
