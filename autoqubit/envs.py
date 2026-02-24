import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .physics import RandomizedPhysics, TripleDotPhysics

class RobustSensingEnv(gym.Env):
    # DQD Environment with Active Sensing and Domain Randomization
    def __init__(self, max_steps=50):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=-10.0, high=10.0, shape=(1, 16, 16), dtype=np.float32),
            "voltage": spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)
        })
        self.physics = RandomizedPhysics()
        self.target_state = np.array([1, 0])
        self.max_steps = max_steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.physics.randomize_device() # Domain Randomization trigger
        self.current_v = np.array([-10.0, -10.0])
        self.steps = 0
        return self._get_obs(0.25), {}

    def _get_obs(self, noise_level):
        scan = self.physics.simulate_sensor_scan(self.current_v[0], self.current_v[1], noise_level)
        return {
            "image": np.expand_dims(scan, axis=0), 
            "voltage": (self.current_v / 20.0).astype(np.float32)
        }

    def step(self, action):
        self.steps += 1
        integration_time = (action[2] + 1.0) / 2.0 
        current_noise = 0.6 - (0.55 * integration_time)
        
        self.current_v += action[0:2] * 2.5 
        obs = self._get_obs(current_noise)
        
        actual_state = self.physics.get_ground_state(*self.current_v)
        reward = -0.1 - (0.4 * integration_time)
        terminated = False
        
        if np.array_equal(actual_state, self.target_state):
            reward += 50.0 
            terminated = True
        elif np.any(np.abs(self.current_v) > 40.0):
            reward -= 10.0
            terminated = True
            
        truncated = self.steps >= self.max_steps
        info = {'integration_time': integration_time, 'noise': current_noise}
        return obs, reward, terminated, truncated, info


class VirtualGateTripleDotEnv(gym.Env):
    # TQD Environment using Virtual Gate linear algebra abstraction
    def __init__(self, max_steps=100):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(6,), dtype=np.float32)
        self.physics = TripleDotPhysics()
        self.target_state = np.array([1, 1, 1]) 
        self.max_steps = max_steps
        
        self.cross_talk_matrix = np.array([
            [1.00, 0.10, 0.01], 
            [0.10, 1.00, 0.10], 
            [0.01, 0.10, 1.00]
        ])
        self.v2p_matrix = np.linalg.inv(self.cross_talk_matrix)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_v = np.array([-15.0, -15.0, -15.0])
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        state = self.physics.get_ground_state(*self.current_v)
        return np.concatenate([self.current_v / 20.0, state]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        physical_action = self.v2p_matrix @ (action * 2.0)
        self.current_v += physical_action 
        
        obs = self._get_obs()
        current_state = obs[3:6]
        
        reward = -0.1
        terminated = False
        
        if np.array_equal(current_state, self.target_state):
            reward += 100.0
            terminated = True
        elif np.any(np.abs(self.current_v) > 40.0):
            reward -= 20.0
            terminated = True
            
        truncated = self.steps >= self.max_steps
        return obs, reward, terminated, truncated, {}
