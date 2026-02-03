#!/usr/bin/env python3
"""
PPO training for RoboCasa PnPCounterToCab with visual observations + R3M visual features (SB3).

What this script does
- Uses RoboCasa (via robosuite) environment: PnPCounterToCab (custom env class recommended below)
- Extracts multi-camera RGB observations
- Uses a *pretrained R3M encoder* as a frozen feature extractor
- Trains PPO (stable-baselines3) on top of those features
- Optionally adds *language-conditioned reward shaping* (dense shaping aligned to instruction)

Notes
- This script assumes your env returns camera images under keys like:
  "robot0_agentview_center_image", etc. (RoboCasa/robosuite convention)
- R3M expects 224x224 RGB, normalized (ImageNet-style). We do:
    uint8 -> float[0,1] -> resize -> normalize -> R3M -> projection MLP

Install dependencies (typical)
- pip install stable-baselines3 gymnasium torch torchvision wandb
- pip install r3m  (or install from the official R3M repo if pip package not available)
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# RoboCasa env import (adjust as needed for your project layout)
# ---------------------------------------------------------------------
# In your uploaded setup, you already used: from env import MyPnPCounterToCab
# Keep that pattern: put custom env class in env.py or similar.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import MyPnPCounterToCab  # noqa: E402

import robosuite  # noqa: F401,E402  (needed to register assets / init)
from robosuite.controllers import load_composite_controller_config  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

try:
    import wandb  # noqa: E402
    from wandb.integration.sb3 import WandbCallback  # noqa: E402
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------
# Utilities: image extraction & wrappers
# ---------------------------------------------------------------------
class VisualWrapper(gym.Wrapper):
    """
    Extract and concatenate multiple camera images from a dict observation.

    Returns an image tensor in torch-friendly shape (C, H, W),
    where C = 3 * num_cameras (RGB per camera concatenated along channel axis).
    """

    def __init__(self, env: gym.Env, camera_basenames: Optional[List[str]] = None):
        super().__init__(env)

        if camera_basenames is None:
            camera_basenames = [
                "robot0_agentview_center",
                "robot0_agentview_left",
                "robot0_agentview_right",
            ]
        self.camera_keys = [f"{name}_image" for name in camera_basenames]

        obs, info = self.env.reset()
        if not isinstance(obs, dict):
            raise TypeError("Expected dict obs from RoboCasa/robosuite env before wrapping.")

        first_img = obs[self.camera_keys[0]]
        if first_img.ndim != 3 or first_img.shape[-1] != 3:
            raise ValueError(f"Expected (H,W,3) image, got {first_img.shape} for {self.camera_keys[0]}.")

        h, w, c = first_img.shape
        n_cam = len(self.camera_keys)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(c * n_cam, h, w),
            dtype=np.uint8,
        )

    def _stack(self, obs_dict: Dict) -> np.ndarray:
        imgs = []
        for k in self.camera_keys:
            if k not in obs_dict:
                raise KeyError(f"Missing key {k}. Available keys: {list(obs_dict.keys())[:30]} ...")
            img = obs_dict[k]  # (H,W,3) uint8
            img_chw = np.transpose(img, (2, 0, 1))
            imgs.append(img_chw)
        return np.concatenate(imgs, axis=0)  # (3*n_cam,H,W)

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._stack(obs_dict), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        return self._stack(obs_dict), reward, terminated, truncated, info


@dataclass
class LangRewardWeights:
    # base dense reward weights
    w_reach: float = 1.0
    w_grasp: float = 3.0
    w_lift: float = 1.0
    w_transport: float = 1.0
    w_place: float = 10.0
    w_success: float = 5.0
    w_action_l2: float = 0.01

    # language-conditioned shaping weights
    w_lang_reach: float = 0.5
    w_lang_transport: float = 0.5
    w_lang_place: float = 1.0


class LanguageRewardShapingWrapper(gym.Wrapper):
    """
    Adds language-conditioned shaping on top of env reward.

    Idea:
    - Parse instruction into "active subgoals"
    - Apply additional dense shaping terms consistent with the text
    - Use *potential-based* style shaping where possible (to reduce bias)

    This wrapper assumes the underlying env exposes (via unwrapped):
    - sim.data.body_xpos[...] for object
    - robots[0].eef_site_id["right"] for ee site position
    - cab.pos for cabinet position
    - check_contact(obj, gripper)
    - _check_success()
    - and OU.obj_inside_of(...) works in the env reward (in your custom env)
    """

    def __init__(self, env: gym.Env, instruction: str, weights: LangRewardWeights):
        super().__init__(env)
        self.instruction = instruction.lower().strip()
        self.w = weights

        # simple keyword-based parsing (robust and fast)
        self.want_pick = any(k in self.instruction for k in ["pick", "grasp", "grab", "take"])
        self.want_place = any(k in self.instruction for k in ["place", "put", "insert", "into"])
        self.want_cabinet = any(k in self.instruction for k in ["cab", "cabinet", "drawer"])
        self.want_counter = "counter" in self.instruction

        # If the instruction is empty, keep shaping minimal
        self.active = any([self.want_pick, self.want_place, self.want_cabinet, self.want_counter])

    def _get_unwrapped(self):
        u = self.env
        while hasattr(u, "env"):
            u = u.env
        # for Monitor, env might be in .unwrapped
        return self.env.unwrapped

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)

        if not self.active:
            return obs, r, terminated, truncated, info

        u = self._get_unwrapped()

        # --- Core quantities ---
        obj_pos = u.sim.data.body_xpos[u.obj_body_id["obj"]]
        ee_pos = u.sim.data.site_xpos[u.robots[0].eef_site_id["right"]]
        dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))
        is_grasped = bool(u.check_contact(u.objects["obj"], u.robots[0].gripper))

        # Many RoboCasa tasks treat cab.pos as cabinet base position (not opening).
        cab_pos = np.array(u.cab.pos, dtype=np.float32)
        dist_obj_cab = float(np.linalg.norm(obj_pos - cab_pos))

        # --- Language-conditioned shaping terms ---
        r_lang = 0.0

        # Encourage reaching object when instruction mentions pick/grasp/take
        if self.want_pick:
            r_lang += self.w.w_lang_reach * np.exp(-2.0 * dist_ee_obj)

        # Encourage transport-to-cabinet only if instruction mentions cabinet/drawer/into
        if (self.want_cabinet or self.want_place) and is_grasped:
            r_lang += self.w.w_lang_transport * np.exp(-2.0 * dist_obj_cab)

        # Encourage completion explicitly if instruction says "into cabinet"
        # We reuse success signal from env when available.
        if (self.want_cabinet or self.want_place) and info.get("is_success", False):
            r_lang += self.w.w_lang_place

        info = dict(info)
        info["reward_lang_shaping"] = float(r_lang)

        return obs, float(r + r_lang), terminated, truncated, info


# ---------------------------------------------------------------------
# R3M feature extractor for SB3
# ---------------------------------------------------------------------
class R3MFeaturesExtractor(BaseFeaturesExtractor):
    """
    Frozen pretrained R3M visual encoder as SB3 feature extractor.

    Input obs: uint8 images shaped (C,H,W) where C = 3*num_cams.

    Processing:
    - split cameras (optional), average features, or encode concatenated (we encode a fused view by averaging cams)
    - resize to 224x224
    - normalize
    - run R3M
    - project to features_dim
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        num_cameras: int = 3,
        r3m_model_name: str = "resnet50",
        freeze_r3m: bool = True,
    ):
        super().__init__(observation_space, features_dim)

        assert len(observation_space.shape) == 3, f"Expected (C,H,W), got {observation_space.shape}"
        c, _, _ = observation_space.shape
        if c % 3 != 0:
            raise ValueError(f"Expected channel multiple of 3, got C={c}")
        self.num_cameras = num_cameras if num_cameras is not None else (c // 3)
        if self.num_cameras != (c // 3):
            # keep consistent; user might set num_cameras explicitly
            self.num_cameras = c // 3

        self.device_for_r3m = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load R3M
        try:
            from r3m import load_r3m  # type: ignore
        except Exception as e:
            raise ImportError(
                "Could not import r3m. Install it first (e.g., `pip install r3m` or from the official repo)."
            ) from e

        self.r3m = load_r3m(r3m_model_name)
        self.r3m.eval()
        self.r3m.to(self.device_for_r3m)

        if freeze_r3m:
            for p in self.r3m.parameters():
                p.requires_grad_(False)

        # Determine R3M output dim by a dummy forward
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224), device=self.device_for_r3m)
            out = self.r3m(dummy)
            r3m_dim = out.shape[-1]

        self.proj = nn.Sequential(
            nn.Linear(r3m_dim, features_dim),
            nn.ReLU(),
        )

        # ImageNet normalization (widely used for ResNet-based encoders)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) uint8 or float
        -> float in [0,1], resized to 224, normalized
        """
        x = x.float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B, C, H, W)
        b, c, h, w = observations.shape
        n_cam = c // 3

        # Split into cameras: (B, n_cam, 3, H, W)
        x = observations.view(b, n_cam, 3, h, w)

        # Encode each camera independently and average features
        feats = []
        for i in range(n_cam):
            xi = x[:, i, :, :, :].to(self.device_for_r3m)
            xi = self._preprocess(xi)
            with torch.set_grad_enabled(self.r3m.training and any(p.requires_grad for p in self.r3m.parameters())):
                fi = self.r3m(xi)  # (B, D)
            feats.append(fi)

        f = torch.stack(feats, dim=0).mean(dim=0)  # (B, D)
        f = f.to(self.proj[0].weight.device)  # ensure same device as proj
        return self.proj(f)


# ---------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------
def make_env(
    rank: int,
    seed: int,
    camera_height: int,
    camera_width: int,
    camera_basenames: List[str],
    instruction: Optional[str],
    lang_weights: LangRewardWeights,
) -> Callable[[], gym.Env]:

    def _init():
        # Offscreen EGL rendering (common for headless training)
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        robots = "PandaOmron"
        controller_config = load_composite_controller_config(controller=None, robot=robots)

        env = MyPnPCounterToCab(
            robots=robots,
            controller_configs=controller_config,
            use_camera_obs=True,
            camera_names=camera_basenames,
            camera_heights=camera_height,
            camera_widths=camera_width,
            has_renderer=False,
            has_offscreen_renderer=True,
            reward_shaping=True,
            control_freq=20,
            ignore_done=False,
            horizon=500,
        )

        env = VisualWrapper(env, camera_basenames=camera_basenames)

        if instruction is not None and instruction.strip():
            env = LanguageRewardShapingWrapper(env, instruction=instruction, weights=lang_weights)

        log_dir = f"/tmp/robocasa_sb3/{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)

        env.reset(seed=seed + rank)
        return env

    return _init


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("SB3 PPO + R3M visual features for RoboCasa PnPCounterToCab")
    parser.add_argument("--max_timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_envs", type=int, default=1)

    parser.add_argument("--camera_height", type=int, default=128)
    parser.add_argument("--camera_width", type=int, default=128)
    parser.add_argument("--cameras", type=str, default="robot0_agentview_center,robot0_agentview_left,robot0_agentview_right")

    # R3M
    parser.add_argument("--r3m_model", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--features_dim", type=int, default=256)
    parser.add_argument("--finetune_r3m", action="store_true", help="If set, do NOT freeze R3M (usually unstable for PPO).")

    # PPO
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)

    # Logging / saving
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="robocasa-ppo-visual")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Language shaping
    parser.add_argument("--instruction", type=str, default="Pick the object from the counter and place it into the cabinet.")
    parser.add_argument("--no_lang_shaping", action="store_true")

    args = parser.parse_args()

    camera_basenames = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if len(camera_basenames) == 0:
        raise ValueError("You must specify at least one camera basename.")

    run_name = args.run_name or f"PnPCounterToCab_ppo_r3m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optional W&B
    wandb_run = None
    callbacks = []

    if args.use_wandb:
        if not _WANDB_AVAILABLE:
            raise RuntimeError("wandb is not installed. Install it or remove --use_wandb.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback(gradient_save_freq=100_000, verbose=0))

    # Checkpointing
    ckpt_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks.append(
        CheckpointCallback(
            save_freq=max(100_000 // max(args.n_envs, 1), 1),
            save_path=ckpt_dir,
            name_prefix="ppo_r3m",
        )
    )
    callback = CallbackList(callbacks) if callbacks else None

    # Env
    lang_weights = LangRewardWeights()
    instruction = None if args.no_lang_shaping else args.instruction

    env = make_vec_env(
        make_env(
            rank=0,
            seed=args.seed,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            camera_basenames=camera_basenames,
            instruction=instruction,
            lang_weights=lang_weights,
        ),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else DummyVecEnv,
    )

    # Policy with custom features extractor
    policy_kwargs = dict(
        features_extractor_class=R3MFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=args.features_dim,
            num_cameras=len(camera_basenames),
            r3m_model_name=args.r3m_model,
            freeze_r3m=not args.finetune_r3m,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        policy="CnnPolicy",  # still OK; we override the extractor
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join("runs", run_name),
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
    )

    model.learn(total_timesteps=args.max_timesteps, callback=callback, progress_bar=True)

    final_path = os.path.join(ckpt_dir, "final_model")
    model.save(final_path)

    env.close()
    if wandb_run is not None:
        wandb.finish()

    print(f"[Done] Saved final model to: {final_path}")


if __name__ == "__main__":
    main()
