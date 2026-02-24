import numpy as np
from scipy.signal import find_peaks

class CrossTalkCalibrator:
    # Automated routine to measure and build the Virtual Gate matrix.
    def __init__(self, physics_engine):
        self.physics = physics_engine
        
    def discover_matrix(self):
        print("Commencing Automated Cross-Talk Calibration...")
        M = np.zeros((3, 3))
        
        for dot_idx in range(3):
            for gate_idx in range(3):
                voltages = np.linspace(-20, 20, 200)
                readings = np.zeros(200)
                current_v = np.array([0.0, 0.0, 0.0])
                
                for i, v in enumerate(voltages):
                    current_v[gate_idx] = v
                    state = self.physics.get_ground_state(*current_v)
                    readings[i] = state[dot_idx]
                    
                peaks, _ = find_peaks(np.gradient(readings), height=0.5)
                
                if len(peaks) > 0 and voltages[peaks[0]] != 0:
                    M[dot_idx, gate_idx] = 1.0 / abs(voltages[peaks[0]])
                else:
                    M[dot_idx, gate_idx] = 0.01 
                
        for i in range(3): 
            M[i, :] = M[i, :] / M[i, i]
            
        print("Discovered Virtual Gate Matrix M:\n", np.round(M, 3))
        return M
