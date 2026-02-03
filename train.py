"""
Training script using the custom PnPCounterToCab environment.

This is a modified version of train.py that uses the custom environment
from the env/ folder, allowing you to experiment with different reward functions.

Usage:
    python train_custom.py --env_name PnPCounterToCab --max_timesteps 500000 --num_envs 2
"""

import sys
import os

# FORCE EGL for offscreen rendering with correct contexts
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Add local skrl to path (repo structure: ./skrl/skrl/...)
skrl_path = os.path.join(os.path.dirname(__file__), 'skrl')
sys.path.insert(0, skrl_path)

# Add local robocasa to path (repo structure: ./robocasa/robocasa/...)
robocasa_path = os.path.join(os.path.dirname(__file__), 'robocasa')
sys.path.insert(0, robocasa_path)

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
import robocasa

# Import custom environment
from env import MyPnPCounterToCab

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import gymnasium
import gymnasium.spaces
import gym.spaces
import numpy as np
import argparse
import yaml

# Define the models (Policies and Values) with CNN Backbone
class CNN(nn.Module):
    def __init__(self, observation_space, features=256):
        super().__init__()
        # Input: (C, H, W) -> e.g., (3, 84, 84)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute output size
        self._obs_shape = observation_space.shape
        with torch.no_grad():
            # Create a dummy observation with correct shape
            dummy = torch.zeros(1, *observation_space.shape)
            out = self.net(dummy)
            self.out_size = out.shape[1]
            
        self.fc = nn.Linear(self.out_size, features)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x might be flattened by SKRL wrapper? [Batch, Flattened]
        # Reshape to (Batch, C, H, W)
        if x.dim() == 2:
            x = x.view(-1, *self._obs_shape)
        
        # Standardize: / 255.0
        x = x / 255.0 
        x = self.net(x)
        x = self.fc(x)
        x = self.relu(x)
        return x

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std, min_log_std=min_log_std, max_log_std=max_log_std, reduction=reduction)

        self.feature_extractor = CNN(observation_space, features=256)
        
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
            nn.Tanh(), 
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # inputs["observations"] is the image tensor
        x = self.feature_extractor(inputs["observations"])
        x = self.net(x)
        return x, {"log_std": self.log_std_parameter}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)

        self.feature_extractor = CNN(observation_space, features=256)

        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        x = self.feature_extractor(inputs["observations"])
        return self.net(x), {}

# Wrapper to extract image from Robosuite dict obs and format for PyTorch (C, H, W)
class RobocasaImageWrapper(gymnasium.Env):
    def __init__(self, env, camera_name="robot0_agentview_center_image", c_heights=128, c_widths=128):
        self.env = env
        self.camera_name = camera_name
        
        # PyTorch expects (C, H, W)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(3, c_heights, c_widths), dtype=np.uint8
        )
        
        # Action space conversion
        if hasattr(env, "action_space"):
            self.action_space = self._convert_space(env.action_space)
        else:
            # Robosuite raw env uses action_spec = (low, high)
            low, high = env.action_spec
            self.action_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
            
        self.metadata = getattr(env, "metadata", {"render_modes": []})
        self.render_mode = getattr(env, "render_mode", None)
        
    def _convert_space(self, space):
        if isinstance(space, gym.spaces.Box):
            return gymnasium.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        elif isinstance(space, gym.spaces.Discrete):
            return gymnasium.spaces.Discrete(n=space.n)
        return space

    def _extract_obs(self, obs_dict):
        # obs_dict contains the image. It is usually (H, W, C) in Robosuite.
        # We assume RGB, so C=3.
        # Robosuite image is often inverted? No, it's usually standard.
        # But PyTorch wants (C, H, W).
        
        image = obs_dict.get(self.camera_name)
        if image is None:
             # Fallback or check keys
             # Sometimes name differs. PnP might use 'agentview_image'
             keys = list(obs_dict.keys())
             # simple heuristic: find first key with 'image'
             img_keys = [k for k in keys if "image" in k]
             if img_keys:
                 image = obs_dict[img_keys[0]]
             else:
                 raise ValueError(f"Could not find image observation. Available keys: {keys}")
        
        # image is (H, W, 3)
        # Transpose to (2, 0, 1)
        image = np.transpose(image, (2, 0, 1))
        return image

    def reset(self, seed=None, options=None):
        if seed is not None and hasattr(self.env, "seed"):
             try:
                self.env.seed(seed)
             except:
                pass
        
        # Robosuite returns a dictionary of observations
        obs_dict = self.env.reset()
        if isinstance(obs_dict, tuple): # Handle if underlying env returns tuple
             obs_dict = obs_dict[0]
             
        obs = self._extract_obs(obs_dict)
        return obs, {}

    def step(self, action):
        # Clip actions to valid range and check for NaN/Inf to prevent simulation instability
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Replace any NaN or Inf values with 0
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        
        ret = self.env.step(action)
        # Handle 4 or 5 values
        if len(ret) == 4:
            obs_dict, reward, done, info = ret
            truncated = False
            terminated = done
        elif len(ret) == 5:
            obs_dict, reward, terminated, truncated, info = ret
        else:
            raise ValueError(f"Unexpected step return length: {len(ret)}")
            
        obs = self._extract_obs(obs_dict)
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    @property
    def unwrapped(self):
        return self.env.unwrapped
        
    def __getattr__(self, name):
        return getattr(self.env, name)


def create_sim_env(env_name, seed=None, c_heights=128, c_widths=128):
    # Setup controller
    robots = "PandaOmron" # Default robot
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots,
    )
    
    # Environment arguments - ENABLE CAMERAS for Visual RL
    env_kwargs = dict(
        robots=robots,
        controller_configs=controller_config,
        has_renderer=False, # Headless training, don't show window
        has_offscreen_renderer=True, # Need this for rendering images for observation
        ignore_done=True,
        use_object_obs=False, # Disable object state obs (Visual RL)
        use_camera_obs=True, # Enable camera obs
        camera_names=["robot0_agentview_center",
                "robot0_agentview_left",
                "robot0_agentview_right"], # Primary camera
        camera_heights=c_heights,
        camera_widths=c_widths,
        control_freq=20,
        seed=seed,
        reward_shaping=True,
    )

    # Use custom environment instead of robosuite.make
    print(f"Creating custom environment: {env_name}")
    env = MyPnPCounterToCab(**env_kwargs)
    
    # DO NOT use GymWrapper here because it flattens everything including images.
    # We use our custom RobocasaImageWrapper directly on the base environment.
    
    # Wrap with Image Wrapper (Robosuite -> Gymnasium Image)
    env = RobocasaImageWrapper(env, camera_name="robot0_agentview_center_image", c_heights=c_heights, c_widths=c_widths)
    
    return env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="PnPCounterToCab", help="Robocasa environment name (for logging)")
    parser.add_argument("--max_timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--headless", action="store_true", help="Just for compat, script forces headless")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    args = parser.parse_args()

    set_seed(args.seed)

    # WandB Configuration from macros_private.yaml (loaded when --wandb is used)
    WANDB_PROJECT = "robocasa_experiments"  # Default project name since args removed
    if args.wandb:
        _macros_path = os.path.join(os.path.dirname(__file__), "macros_private.yaml")
        with open(_macros_path) as f:
            _macros = yaml.safe_load(f)
        WANDB_ENTITY = _macros["WANDB_ENTITY"]
        WANDB_API_KEY = _macros["WANDB_API_KEY"]
        import wandb
        wandb.login(key=WANDB_API_KEY)

    # 1. Create environment(s)
    # Wrapper helper to allow pickling if needed, though lambda is usually fine with AsyncVectorEnv
    def make_env(rank):
        def _thunk():
            # Force EGL again just in case (though env var should handle it)
            # os.environ["MUJOCO_GL"] = "egl" 
            env = create_sim_env(args.env_name, seed=args.seed + rank)
            return env
        return _thunk

    if args.num_envs > 1:
        # Vectorized environment
        print(f"Creating {args.num_envs} parallel environments...")
        # Robocasa is CPU heavy, so using AsyncVectorEnv (multiprocessing)
        # CRITICAL: Use 'spawn' for CUDA/OpenGL compatibility
        # Gymnasium expects context name as string
        env = gymnasium.vector.AsyncVectorEnv(
            [make_env(i) for i in range(args.num_envs)], 
            context="spawn"
        )
    else:
        # Single environment
        env = create_sim_env(args.env_name, seed=args.seed)
    
    # 2. Wrap for SKRL
    # SKRL wrap_env automatically handles gymnasium.vector.VectorEnv
    env = wrap_env(env)
    
    # Force agent device to GPU if available, even if env is CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Agent Device: {device}")

    # 3. Instantiate memory
    memory = RandomMemory(memory_size=10000, num_envs=env.num_envs, device=device)

    # 4. Instantiate models
    # Note: SKRL models will automatically move to device
    models = {}
    models["policy"] = Policy(env.observation_space, env.state_space, env.action_space, device)
    models["value"] = Value(env.observation_space, env.state_space, env.action_space, device)

    # 5. Configure PPO
    cfg = PPO_CFG()
    # Adjust rollouts logic:
    # If single env, 2048 steps per update.
    # If 10 envs, 2048 steps per env -> 20480 total steps. That's a lot.
    # Usually we want Total Steps per update = num_envs * rollouts.
    # If we want ~2048 total steps per update:
    target_total_steps = 2048
    cfg.rollouts = max(32, target_total_steps // args.num_envs)

    cfg.learning_epochs = 10
    cfg.mini_batches = 32 # mini_batches is number of minibatches to split buffer into
    cfg.discount_factor = 0.9
    cfg.lambda_ = 0.95
    cfg.learning_rate = 5e-4
    cfg.learning_rate_scheduler = KLAdaptiveLR
    cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.008}
    cfg.grad_norm_clip = 0.5
    cfg.ratio_clip = 0.2
    cfg.value_clip = 0.2
    cfg.entropy_loss_scale = 0.0
    cfg.value_loss_scale = 0.5
    
    # Use standard scaler? For images, we usually just divide by 255.
    # The CNN class handles / 255.0. 
    # disable standard scaler for obs
    cfg.observation_preprocessor = None 
    
    # Logging
    cfg.experiment.directory = f"runs/robocasa/{args.env_name}_custom_visual"
    cfg.experiment.write_interval = 200
    cfg.experiment.checkpoint_interval = 2000
    
    # Configure WandB
    if args.wandb:
        cfg.experiment.wandb = True
        cfg.experiment.wandb_kwargs = {
            "project": WANDB_PROJECT,
            "entity": WANDB_ENTITY,
            "name": f"robocasa_{args.env_name}_custom",
            "monitor_gym": True,
            "sync_tensorboard": True  # Enable tensorboard syncing to wandb
    }

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
    )

    # 6. Trainer
    cfg_trainer = {"timesteps": args.max_timesteps, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # 7. Train
    print(f"Starting VISUAL training on CUSTOM {args.env_name} with {args.num_envs} environments...")
    print(f"Using custom reward function from env/custom_pnp_counter_to_cab.py")
    trainer.train()
