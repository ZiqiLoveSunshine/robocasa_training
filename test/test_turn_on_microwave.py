"""
Test script for MyTurnOnMicrowave custom environment.

This script tests that the custom environment can be instantiated
and that the reward function works correctly.
"""

import sys
import os

# Add env directory to path
env_path = os.path.join(os.path.dirname(__file__), 'env')
if env_path not in sys.path:
    sys.path.insert(0, env_path)

from custom_turn_on_microwave import MyTurnOnMicrowave
import numpy as np


def test_environment_creation():
    """Test that the environment can be created successfully."""
    print("=" * 70)
    print("Testing MyTurnOnMicrowave Environment Creation")
    print("=" * 70)
    
    try:
        env = MyTurnOnMicrowave(
            robots="PandaOmron",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            horizon=200,
        )
        print("✓ Environment created successfully!")
        return env
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_reward_function(env):
    """Test that the reward function works correctly."""
    print("\n" + "=" * 70)
    print("Testing Reward Function")
    print("=" * 70)
    
    try:
        # Reset environment
        obs = env.reset()
        print("✓ Environment reset successfully!")
        
        # Test reward calculation
        action = np.zeros(env.action_dim)
        reward = env.reward(action)
        print(f"✓ Initial reward: {reward:.4f}")
        
        # Take a few steps
        print("\nTaking 5 random steps...")
        for i in range(5):
            action = env.action_spec[0] + np.random.randn(env.action_dim) * 0.1
            obs, reward, done, info = env.step(action)
            print(f"  Step {i+1}: reward = {reward:.4f}, done = {done}")
            
            if done:
                print("  Episode finished!")
                break
        
        print("\n✓ Reward function works correctly!")
        return True
        
    except Exception as e:
        print(f"\n✗ Reward function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_components(env):
    """Test individual reward components."""
    print("\n" + "=" * 70)
    print("Testing Reward Components")
    print("=" * 70)
    
    try:
        env.reset()
        
        # Get initial state
        gripper_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        button_id = env.sim.model.geom_name2id(
            "{}start_button".format(env.microwave.naming_prefix)
        )
        button_pos = env.sim.data.geom_xpos[button_id]
        dist = np.linalg.norm(gripper_pos - button_pos)
        turned_on = env.microwave.get_state()["turned_on"]
        
        print(f"Initial state:")
        print(f"  Gripper position: {gripper_pos}")
        print(f"  Button position: {button_pos}")
        print(f"  Distance to button: {dist:.4f} m")
        print(f"  Microwave turned on: {turned_on}")
        
        # Test reward with zero action
        action = np.zeros(env.action_dim)
        reward = env.reward(action)
        print(f"\nReward with zero action: {reward:.4f}")
        
        # Test reward with large action (should have penalty)
        large_action = np.ones(env.action_dim) * 0.5
        reward_large = env.reward(large_action)
        print(f"Reward with large action: {reward_large:.4f}")
        print(f"Action penalty: {reward - reward_large:.4f}")
        
        print("\n✓ Reward components test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Reward components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("\n" + "=" * 70)
    print("MyTurnOnMicrowave Custom Environment Test Suite")
    print("=" * 70 + "\n")
    
    # Test 1: Environment creation
    env = test_environment_creation()
    if env is None:
        print("\n✗ Tests failed: Could not create environment")
        return
    
    # Test 2: Reward function
    if not test_reward_function(env):
        print("\n✗ Tests failed: Reward function error")
        return
    
    # Test 3: Reward components
    if not test_reward_components(env):
        print("\n✗ Tests failed: Reward components error")
        return
    
    # All tests passed
    print("\n" + "=" * 70)
    print("✓ All tests passed successfully!")
    print("=" * 70)
    print("\nYou can now use MyTurnOnMicrowave in your training scripts.")
    print("Example usage:")
    print("  from env.custom_turn_on_microwave import MyTurnOnMicrowave")
    print("  env = MyTurnOnMicrowave(robots='PandaOmron', ...)")


if __name__ == "__main__":
    main()
