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
import robocasa

# Import robocasa environments
try:
    import robocasa.environments.kitchen.single_stage.kitchen_pnp
except ImportError:
    pass

# Setup controller
robots = "PandaOmron"
controller_config = load_composite_controller_config(
    controller=None,
    robot=robots,
)

# Create environment with cameras enabled
env_kwargs = dict(
    env_name="PnPCounterToCab",
    robots=robots,
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_object_obs=False,
    use_camera_obs=True,
    camera_names=["robot0_agentview_center"],  # Try to get this camera
    camera_heights=128,
    camera_widths=128,
    control_freq=20,
    seed=42,
    reward_shaping=True,
)

print("Creating environment...")
env = robosuite.make(**env_kwargs)

print("\nResetting environment...")
obs = env.reset()

print("\n" + "="*60)
print("OBSERVATION KEYS:")
print("="*60)
for key in sorted(obs.keys()):
    if isinstance(obs[key], (list, tuple)):
        print(f"  {key}: {type(obs[key])} (length: {len(obs[key])})")
    elif hasattr(obs[key], 'shape'):
        print(f"  {key}: shape={obs[key].shape}, dtype={obs[key].dtype}")
    else:
        print(f"  {key}: {type(obs[key])}")

print("\n" + "="*60)
print("CAMERA-RELATED KEYS:")
print("="*60)
camera_keys = [k for k in obs.keys() if 'image' in k.lower() or 'camera' in k.lower()]
for key in sorted(camera_keys):
    if hasattr(obs[key], 'shape'):
        print(f"  {key}: shape={obs[key].shape}, dtype={obs[key].dtype}")
    else:
        print(f"  {key}: {obs[key]}")

print("\n" + "="*60)
print("ENVIRONMENT INFO:")
print("="*60)
print(f"Camera names requested: {env_kwargs['camera_names']}")
print(f"Camera heights: {env_kwargs['camera_heights']}")
print(f"Camera widths: {env_kwargs['camera_widths']}")

env.close()
print("\nDone!")
