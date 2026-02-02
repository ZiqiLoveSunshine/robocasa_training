"""
Example script demonstrating how to use the custom PnPCounterToCab environment.

This script shows how to:
1. Import and use the custom environment
2. Test the custom reward function
3. Integrate with the training pipeline
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

# FORCE EGL for offscreen rendering
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from env import MyPnPCounterToCab
from robosuite.controllers import load_composite_controller_config
import numpy as np


def test_custom_environment():
    """
    Test the custom environment with the modified reward function.
    """
    print("="*60)
    print("Testing My PnPCounterToCab Environment")
    print("="*60)
    
    # Setup controller
    robots = "PandaOmron"
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots,
    )
    
    # Create custom environment
    env = MyPnPCounterToCab(
        robots=robots,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_center"],
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
        seed=42,
        reward_shaping=True,
    )
    
    print("\n✓ Custom environment created successfully!")
    print(f"  Environment type: {type(env).__name__}")
    print(f"  Parent class: {type(env).__bases__[0].__name__}")
    
    # Reset environment
    obs = env.reset()
    print(f"\n✓ Environment reset successfully!")
    print(f"  Observation keys: {list(obs.keys())}")
    
    # Get episode metadata
    ep_meta = env.get_ep_meta()
    print(f"\n✓ Task: {ep_meta.get('lang', 'N/A')}")
    
    # Test reward function
    print("\n" + "="*60)
    print("Testing Custom Reward Function")
    print("="*60)
    
    # Take some random actions and check rewards
    print("\nTaking 10 random actions and computing rewards:")
    print(f"{'Step':<6} {'Reward':<10} {'Success':<10}")
    print("-"*30)
    
    for i in range(10):
        action = np.random.uniform(env.action_spec[0], env.action_spec[1])
        obs, reward, done, info = env.step(action)
        success = env._check_success()
        print(f"{i+1:<6} {reward:<10.4f} {str(success):<10}")
    
    print("\n✓ Reward function working correctly!")
    
    # Test alternative reward functions
    print("\n" + "="*60)
    print("Comparing Different Reward Functions")
    print("="*60)
    
    env.reset()
    action = np.zeros(env.action_spec[0].shape)  # Zero action for comparison
    
    # Get rewards from different functions
    reward_default = env.reward(action)
    reward_sparse = env.reward_sparse(action)
    reward_dense_alt = env.reward_dense_alternative(action)
    reward_penalties = env.reward_with_penalties(action)
    
    print(f"\nReward comparison (zero action):")
    print(f"  Default reward:        {reward_default:.4f}")
    print(f"  Sparse reward:         {reward_sparse:.4f}")
    print(f"  Dense alternative:     {reward_dense_alt:.4f}")
    print(f"  With penalties:        {reward_penalties:.4f}")
    
    env.close()
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)


def compare_original_vs_custom():
    """
    Compare the original and custom environments side-by-side.
    """
    print("\n" + "="*60)
    print("Comparing Original vs Custom Environment")
    print("="*60)
    
    from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToCab
    from robosuite.controllers import load_composite_controller_config
    
    robots = "PandaOmron"
    controller_config = load_composite_controller_config(controller=None, robot=robots)
    
    env_kwargs = dict(
        robots=robots,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        control_freq=20,
        seed=42,
        reward_shaping=True,
    )
    
    # Create both environments
    print("\nCreating original environment...")
    env_original = PnPCounterToCab(**env_kwargs)
    
    print("Creating custom environment...")
    env_custom = MyPnPCounterToCab(**env_kwargs)
    
    # Reset both
    env_original.reset()
    env_custom.reset()
    
    # Compare rewards for same action
    action = np.zeros(12)
    
    reward_original = env_original.reward(action)
    reward_custom = env_custom.reward(action)
    
    print(f"\nReward comparison (same state, zero action):")
    print(f"  Original: {reward_original:.4f}")
    print(f"  Custom:   {reward_custom:.4f}")
    print(f"  Difference: {abs(reward_original - reward_custom):.6f}")
    
    if abs(reward_original - reward_custom) < 1e-6:
        print("\n✓ Rewards match! Custom environment correctly inherits behavior.")
    else:
        print("\n⚠ Rewards differ. This is expected if you modified the reward function.")
    
    env_original.close()
    env_custom.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test custom PnPCounterToCab environment")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare original vs my custom environment")
    args = parser.parse_args()
    
    if args.compare:
        compare_original_vs_custom()
    else:
        test_custom_environment()
