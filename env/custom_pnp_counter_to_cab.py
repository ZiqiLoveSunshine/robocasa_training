"""
Custom PnPCounterToCab environment with modified reward function.

This class inherits from the original RoboCasa PnPCounterToCab environment
and allows you to customize the reward function without modifying the
original robocasa or skrl packages.

Usage:
    from env import MyPnPCounterToCab
    
    env = MyPnPCounterToCab(
        robots="PandaOmron",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_center"],
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
        reward_shaping=True,
    )
"""

import sys
import os

# Add robocasa to path if not already there
robocasa_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'robocasa')
if robocasa_path not in sys.path:
    sys.path.insert(0, robocasa_path)

from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToCab
import robocasa.utils.object_utils as OU
import numpy as np


class MyPnPCounterToCab(PnPCounterToCab):
    """
    My custom PnPCounterToCab environment with modified reward function.
    
    This class inherits from the original PnPCounterToCab and overrides
    the reward() method to implement a custom reward function.
    
    You can modify the reward function below to experiment with different
    reward shaping strategies without touching the original robocasa code.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the custom environment.
        All arguments are passed to the parent PnPCounterToCab class.
        """
        super().__init__(*args, **kwargs)
        
    def reward(self, action=None):
        """
        Using alternative dense reward with different shaping.
        """
        reward = 0.0
        
        obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        
        # Distance-based rewards with different scaling
        dist_gripper_obj = np.linalg.norm(gripper_site_pos - obj_pos)
        dist_obj_cab = np.linalg.norm(obj_pos - self.cab.pos)
        
        # Exponential decay rewards
        reward += np.exp(-2.0 * dist_gripper_obj)  # Reach object
        
        # Check if grasped
        is_grasped = self.check_contact(self.objects["obj"], self.robots[0].gripper)
        if is_grasped:
            reward += 3.0  # Smaller grasp bonus
            reward += np.exp(-2.0 * dist_obj_cab)  # Only reward moving to cab if grasped
        
        # Place bonus
        if OU.obj_inside_of(self, "obj", self.cab):
            reward += 10.0  # Larger place bonus
            
        # Success bonus
        if self._check_success():
            reward += 5.0
            
        return reward
    
    # ============================================================================
    # EXAMPLE ALTERNATIVE REWARD FUNCTIONS
    # ============================================================================
    # Below are some example alternative reward functions you can use.
    # To use one, simply rename it to 'reward' and rename the current 'reward'
    # to something else (e.g., 'reward_original').
    
    def reward_sparse(self, action=None):
        """
        Sparse reward: only reward on task completion.
        This is much harder to learn but avoids reward shaping bias.
        """
        if self._check_success():
            return 10.0
        return 0.0
    
    def reward_dense_alternative(self, action=None):
        """
        Alternative dense reward with different shaping.
        """
        reward = 0.0
        
        obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        
        # Distance-based rewards with different scaling
        dist_gripper_obj = np.linalg.norm(gripper_site_pos - obj_pos)
        dist_obj_cab = np.linalg.norm(obj_pos - self.cab.pos)
        
        # Exponential decay rewards
        reward += np.exp(-2.0 * dist_gripper_obj)  # Reach object
        
        # Check if grasped
        is_grasped = self.check_contact(self.objects["obj"], self.robots[0].gripper)
        if is_grasped:
            reward += 3.0  # Smaller grasp bonus
            reward += np.exp(-2.0 * dist_obj_cab)  # Only reward moving to cab if grasped
        
        # Place bonus
        if OU.obj_inside_of(self, "obj", self.cab):
            reward += 10.0  # Larger place bonus
            
        # Success bonus
        if self._check_success():
            reward += 5.0
            
        return reward
    
    def reward_curriculum(self, action=None):
        """
        Curriculum-based reward that changes based on episode count.
        You would need to track episode count externally.
        """
        # This is a template - you'd need to add episode tracking
        reward = 0.0
        
        obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        
        dist_gripper_obj = np.linalg.norm(gripper_site_pos - obj_pos)
        dist_obj_cab = np.linalg.norm(obj_pos - self.cab.pos)
        
        # Always encourage reaching
        reward += 1 - np.tanh(5.0 * dist_gripper_obj)
        
        # Grasp bonus
        is_grasped = self.check_contact(self.objects["obj"], self.robots[0].gripper)
        if is_grasped:
            reward += 5.0
            reward += 1 - np.tanh(5.0 * dist_obj_cab)
        
        # Placement
        if OU.obj_inside_of(self, "obj", self.cab):
            reward += 5.0
            
        if self._check_success():
            reward += 2.0
            
        return reward
    
    def reward_with_penalties(self, action=None):
        """
        Reward function with penalties for undesired behaviors.
        """
        reward = 0.0
        
        obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        
        # Standard rewards
        dist_gripper_obj = np.linalg.norm(gripper_site_pos - obj_pos)
        r_reach_obj = 1 - np.tanh(5.0 * dist_gripper_obj)
        reward += r_reach_obj
        
        is_grasped = self.check_contact(self.objects["obj"], self.robots[0].gripper)
        if is_grasped:
            reward += 5.0
            
        dist_obj_cab = np.linalg.norm(obj_pos - self.cab.pos)
        r_reach_cab = 1 - np.tanh(5.0 * dist_obj_cab)
        reward += r_reach_cab
        
        is_placed = OU.obj_inside_of(self, "obj", self.cab)
        if is_placed:
            reward += 5.0

        if self._check_success():
            reward += 2.0
        
        # Penalties
        # Penalize large actions (encourage smooth movements)
        if action is not None:
            action_penalty = 0.01 * np.sum(np.square(action))
            reward -= action_penalty
        
        # Penalize touching distractor objects
        if self.check_contact(self.objects["distr_counter"], self.robots[0].gripper):
            reward -= 0.5
        if self.check_contact(self.objects["distr_cab"], self.robots[0].gripper):
            reward -= 0.5
            
        return reward


# Example of how to register this environment with robosuite if needed
def register_custom_env():
    """
    Register the custom environment with robosuite.
    This allows you to use robosuite.make("MyPnPCounterToCab", ...)
    """
    import robosuite
    from robosuite.environments.base import register_env
    
    register_env(MyPnPCounterToCab)
