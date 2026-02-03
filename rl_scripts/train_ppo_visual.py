"""
PPO Training Script for Robocasa PnPCounterToCab with Visual Input
Uses multi-camera visual observations from robot cameras.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Callable

# Add parent directory to path to import env module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import MyPnPCounterToCab
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import robosuite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import wandb
from wandb.integration.sb3 import WandbCallback


class MultiCameraCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for multi-camera visual input.
    Processes images from multiple cameras and concatenates their features.
    
    :param observation_space: The observation space (should be a Box space with image data)
    :param features_dim: Number of features extracted (output dimension)
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Assuming observation_space.shape is (C, H, W) where C = 3 * num_cameras
        # For 3 cameras with RGB: (9, H, W)
        n_input_channels = observation_space.shape[0]
        
        # CNN architecture for each camera (Nature DQN architecture)
        # We'll process all cameras together since they're concatenated in channel dimension
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample / 255.0).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize pixel values to [0, 1]
        normalized_obs = observations.float() / 255.0
        features = self.cnn(normalized_obs)
        return self.linear(features)


class VisualWrapper(gym.Wrapper):
    """
    Wrapper to extract and concatenate multiple camera images from Robosuite observations.
    Converts dict observations to a single stacked image tensor.
    """
    
    def __init__(self, env, camera_names=None):
        super().__init__(env)
        
        if camera_names is None:
            self.camera_names = [
                "robot0_agentview_center_image",
                "robot0_agentview_left_image", 
                "robot0_agentview_right_image"
            ]
        else:
            self.camera_names = [f"{cam}_image" for cam in camera_names]
        
        # Get a sample observation to determine image shape
        sample_obs = env.reset()
        if isinstance(sample_obs, tuple):
            sample_obs = sample_obs[0]
        
        # Get image shape from first camera
        first_img = sample_obs[self.camera_names[0]]
        h, w, c = first_img.shape
        
        # New observation space: stack all cameras in channel dimension
        # (H, W, C) * num_cameras -> (C * num_cameras, H, W) in PyTorch format
        num_cameras = len(self.camera_names)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(c * num_cameras, h, w),
            dtype=np.uint8
        )
        
    def _extract_images(self, obs_dict):
        """Extract and concatenate images from observation dictionary."""
        images = []
        for cam_name in self.camera_names:
            if cam_name not in obs_dict:
                raise ValueError(f"Camera {cam_name} not found in observation. Available keys: {obs_dict.keys()}")
            img = obs_dict[cam_name]  # Shape: (H, W, C)
            # Convert from (H, W, C) to (C, H, W) for PyTorch
            img = np.transpose(img, (2, 0, 1))
            images.append(img)
        
        # Concatenate along channel dimension: (C*num_cameras, H, W)
        stacked = np.concatenate(images, axis=0)
        return stacked
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs_dict, info = obs
        else:
            obs_dict = obs
            info = {}
        
        visual_obs = self._extract_images(obs_dict)
        return visual_obs, info
    
    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        visual_obs = self._extract_images(obs_dict)
        return visual_obs, reward, terminated, truncated, info


def make_env(task_name: str, rank: int, seed: int = 0, camera_height: int = 128, 
             camera_width: int = 128) -> Callable:
    """
    Utility function for multiprocessed env with visual observations.
    
    :param task_name: the task class name
    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    :param camera_height: height of camera images
    :param camera_width: width of camera images
    """
    def _init():
        # Force EGL for offscreen rendering
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        
        # Define Environment
        if task_name == "PnPCounterToCab":
            env_cls = MyPnPCounterToCab
        else:
            raise ValueError(f"Task {task_name} not supported yet in this script")

        robots = "PandaOmron"  # Default robot
        controller_config = load_composite_controller_config(
            controller=None,
            robot=robots,
        )
        
        camera_names = [
            "robot0_agentview_center",
            "robot0_agentview_left",
            "robot0_agentview_right"
        ]
        
        env = env_cls(
            robots=robots,
            controller_configs=controller_config,
            use_camera_obs=True,  # Enable camera observations
            camera_names=camera_names,
            camera_heights=camera_height,
            camera_widths=camera_width,
            has_renderer=False,  # No on-screen rendering
            has_offscreen_renderer=True,  # Enable offscreen rendering for cameras
            reward_shaping=True,
            control_freq=20,
            ignore_done=False,
            horizon=500,
        )
        
        # Wrap with VisualWrapper to extract images
        env = VisualWrapper(env, camera_names=camera_names)
        
        # Wrap with Monitor for SB3 logging
        log_dir = f"/tmp/gym_visual/{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="PPO Training for Robocasa with Visual Input")
    parser.add_argument("--task", type=str, default="PnPCounterToCab", help="Task name")
    parser.add_argument("--max_timesteps", type=int, default=1000000, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb-project", type=str, default="Graph-Mobile-Manipulator", help="WandB Project Name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB Entity")
    parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--run_name", type=str, default=None, help="Run name")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--camera_height", type=int, default=128, help="Camera image height")
    parser.add_argument("--camera_width", type=int, default=128, help="Camera image width")
    parser.add_argument("--features_dim", type=int, default=512, help="CNN feature dimension")
    
    args = parser.parse_args()
    
    run_name = args.run_name if args.run_name else f"{args.task}_ppo_visual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize WandB
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    
    # Create Vectorized Environment
    print(f"Creating {args.n_envs} environment(s) with visual observations...")
    env = make_vec_env(
        make_env(args.task, 0, args.seed, args.camera_height, args.camera_width),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    )

    # Define custom policy with CNN feature extractor
    policy_kwargs = dict(
        features_extractor_class=MultiCameraCNNExtractor,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # Policy and value network architectures
    )

    # Define PPO Model with CnnPolicy
    print("Initializing PPO with visual policy...")
    model = PPO(
        "CnnPolicy",  # Use CNN policy for image observations
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"runs/{run_name}",
        learning_rate=3e-4,  # Slightly lower LR for visual learning
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Small entropy bonus for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=os.path.join(args.save_dir, run_name),
        name_prefix="ppo_visual_robocasa",
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=100_000,
        model_save_path=os.path.join(args.save_dir, run_name, "wandb_models"),
        verbose=2,
    )
    
    callbacks = CallbackList([checkpoint_callback, wandb_callback])

    print(f"Starting visual training for {args.max_timesteps} steps...")
    print(f"Using cameras: robot0_agentview_center, robot0_agentview_left, robot0_agentview_right")
    print(f"Image size: {args.camera_height}x{args.camera_width}")
    print(f"Device: {model.device}")
    
    model.learn(
        total_timesteps=args.max_timesteps, 
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, run_name, "final_model")
    model.save(final_model_path)
    print(f"Training finished. Final model saved to: {final_model_path}")
    
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
