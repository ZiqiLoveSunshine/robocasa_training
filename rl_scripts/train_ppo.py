"""
PPO Training Script for Robocasa PnPCounterToCab
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Add parent directory to path to import env module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import MyPnPCounterToCab
from env import MyTurnOnMicrowave
import gymnasium as gym
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from env.fixed_observation_wrapper import FixedObservationWrapper
import wandb
from wandb.integration.sb3 import WandbCallback

def make_env(task_name, rank, seed=0, wandb_run=None):
    """
    Utility function for multiprocessed env.
    
    :param task_name: the task class name
    :param rank: index of the subprocess
    :param seed: the inital seed for RNG
    """
    def _init():
        # Define Environment
        if task_name == "PnPCounterToCab":
            env_cls = MyPnPCounterToCab
        elif task_name == "TurnOnMicrowave":
            env_cls = MyTurnOnMicrowave
        else:
             raise ValueError(f"Task {task_name} not supported yet in this script")

        robots = "PandaOmron" # Default robot
        controller_config = load_composite_controller_config(
            controller=None,
            robot=robots,
        )
        env = env_cls(
            robots=robots,
            controller_configs=controller_config,
            use_camera_obs=False, 
            has_renderer=False, 
            has_offscreen_renderer=False,
            reward_shaping=True, 
            control_freq=20,
            renderer="mjviewer",
            render_camera="robot0_robotview", # avoiding 'robot0_agentview_center' error
            ignore_done=False, 
            horizon=500,
        )
        
        # Wrap with GymWrapper
        # NOTE: Using keys=None due to a bug in RoboCasa's GymWrapper where explicit keys
        # cause observation space to not match actual observations. With keys=None, the
        # wrapper correctly handles the observation space.
        env = GymWrapper(env, keys=None)
        
        # Fix observation space mismatch (RoboCasa bug workaround)
        # env = FixedObservationWrapper(env)
        
        # Wrap with Monitor for SB3 logging
        log_dir = f"/tmp/gym/{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="PPO Training for Robocasa")
    parser.add_argument("--task", type=str, default="PnPCounterToCab", help="Task name")
    parser.add_argument("--max_timesteps", type=int, default=1000000, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb-project", type=str, default="Graph-Mobile-Manipulator", help="WandB Project Name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB Entity")
    parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--run_name", type=str, default=None, help="Run name")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    
    args = parser.parse_args()
    
    run_name = args.run_name if args.run_name else f"{args.task}_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    # Create a list of environment factories, one for each parallel environment
    env_fns = [make_env(args.task, i, args.seed) for i in range(args.n_envs)]
    
    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # Define PPO Model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run_name}",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.9,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=os.path.join(args.save_dir, run_name),
        name_prefix="ppo",
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=100_000,
        model_save_path=os.path.join(args.save_dir, run_name, "wandb_models"),
        verbose=2,
    )
    
    callbacks = CallbackList([checkpoint_callback, wandb_callback])

    print(f"Starting training for {args.max_timesteps} steps...")
    model.learn(
        total_timesteps=args.max_timesteps, 
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    model.save(os.path.join(args.save_dir, run_name, "final_model"))
    print("Training finished.")
    
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()
