"""Debug script to check observation space"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import MyTurnOnMicrowave
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(rank=0, seed=42):
    def _init():
        controller_config = load_composite_controller_config(controller=None, robot="PandaOmron")
        env = MyTurnOnMicrowave(
            robots="PandaOmron",
            controller_configs=controller_config,
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            reward_shaping=True,
            control_freq=20,
            horizon=500,
        )
        
        # Wrap with GymWrapper
        obs_keys = ["robot0_proprio-state", "object-state"]
        env = GymWrapper(env, keys=obs_keys)
        
        # Wrap with Monitor
        log_dir = f"/tmp/gym/{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        
        env.reset(seed=seed + rank)
        return env
    return _init

# Test single environment
print("Testing single environment...")
env_fn = make_env(0, 42)
env = env_fn()
print(f"Observation space: {env.observation_space.shape}")
obs, _ = env.reset()
print(f"Reset observation shape: {obs.shape}")

for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {i+1}: obs shape = {obs.shape}, done = {done}")
    if done or truncated:
        obs, _ = env.reset()
        print(f"  After reset: obs shape = {obs.shape}")

print("\n✓ Single environment works!")

# Test vectorized environment
print("\nTesting vectorized environment...")
env_fns = [make_env(i, 42) for i in range(2)]
vec_env = DummyVecEnv(env_fns)
print(f"Vec observation space: {vec_env.observation_space.shape}")
obs = vec_env.reset()
print(f"Vec reset observation shape: {obs.shape}")

for i in range(5):
    actions = [vec_env.action_space.sample() for _ in range(2)]
    obs, rewards, dones, infos = vec_env.step(actions)
    print(f"Vec step {i+1}: obs shape = {obs.shape}")

print("\n✓ Vectorized environment works!")
