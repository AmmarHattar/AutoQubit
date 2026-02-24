import os
from stable_baselines3 import SAC
from autoqubit.envs import RobustSensingEnv, VirtualGateTripleDotEnv

os.makedirs('models', exist_ok=True)

def train_dqd():
    print("Training Domain-Randomized DQD Agent...")
    env = RobustSensingEnv()
    model = SAC("MultiInputPolicy", env, verbose=1, learning_rate=1e-3, batch_size=64)
    model.learn(total_timesteps=30000)
    model.save("models/sac_robust_dqd_tuner")
    print("Saved to 'models/sac_robust_dqd_tuner.zip'")

def train_tqd():
    print("\nTraining Virtual Gate TQD Agent...")
    env = VirtualGateTripleDotEnv()
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3, batch_size=128)
    model.learn(total_timesteps=20000)
    model.save("models/sac_virtual_tqd")
    print("Saved to 'models/sac_virtual_tqd.zip'")

if __name__ == "__main__":
    train_dqd()
    train_tqd() 
