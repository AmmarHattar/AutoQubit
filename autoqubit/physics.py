import numpy as np

class RandomizedPhysics:
    # Physics engine for a Double Quantum Dot with Domain Randomization.
    def __init__(self):
        self.randomize_device()

    def randomize_device(self):
        self.E_c1 = np.random.uniform(0.8, 1.2)
        self.E_c2 = np.random.uniform(0.8, 1.2)
        self.E_m = np.random.uniform(0.1, 0.4)
        self.alpha_1 = np.random.uniform(0.03, 0.08)
        self.alpha_2 = np.random.uniform(0.03, 0.08)
        self.states = np.array([[n1, n2] for n1 in range(4) for n2 in range(4)])

    def get_ground_state(self, v1, v2):
        ng1, ng2 = v1 * self.alpha_1, v2 * self.alpha_2
        energies = (self.E_c1 * (self.states[:, 0] - ng1)**2 +
                    self.E_c2 * (self.states[:, 1] - ng2)**2 +
                    self.E_m * (self.states[:, 0] - ng1) * (self.states[:, 1] - ng2))
        return self.states[np.argmin(energies)]

    def simulate_sensor_scan(self, v1_center, v2_center, noise_level, window=10.0, res=16):
        v1_arr = np.linspace(v1_center - window/2, v1_center + window/2, res)
        v2_arr = np.linspace(v2_center - window/2, v2_center + window/2, res)
        V1, V2 = np.meshgrid(v1_arr, v2_arr)
        
        charge_map = np.zeros((res, res))
        for i in range(res):
            for j in range(res):
                state = self.get_ground_state(V1[i, j], V2[i, j])
                charge_map[i, j] = state[0] + state[1] 
                
        grad_y, grad_x = np.gradient(charge_map)
        signal = np.sqrt(grad_x**2 + grad_y**2)
        
        noise = np.random.normal(0, noise_level, size=(res, res))
        return (signal + noise).astype(np.float32)


class TripleDotPhysics:
    # Physics engine for a Triple Quantum Dot Array.
    def __init__(self):
        self.E_c = np.array([1.0, 1.0, 1.0]) 
        self.E_m = np.array([0.25, 0.25, 0.05]) 
        self.alpha = np.array([0.05, 0.05, 0.05])
        self.states = np.array([[n1, n2, n3] for n1 in range(3) for n2 in range(3) for n3 in range(3)])

    def get_ground_state(self, v1, v2, v3):
        n_g = np.array([v1, v2, v3]) * self.alpha 
        energies = (
            self.E_c[0]*(self.states[:,0]-n_g[0])**2 + 
            self.E_c[1]*(self.states[:,1]-n_g[1])**2 + 
            self.E_c[2]*(self.states[:,2]-n_g[2])**2 +
            self.E_m[0]*(self.states[:,0]-n_g[0])*(self.states[:,1]-n_g[1]) + 
            self.E_m[1]*(self.states[:,1]-n_g[1])*(self.states[:,2]-n_g[2]) + 
            self.E_m[2]*(self.states[:,0]-n_g[0])*(self.states[:,2]-n_g[2])
        )
        return self.states[np.argmin(energies)]
