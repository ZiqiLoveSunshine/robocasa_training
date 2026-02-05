"""
Custom TurnOnMicrowave environment with modified reward function.

This class inherits from the original RoboCasa TurnOnMicrowave environment
and allows you to customize the reward function without modifying the
original robocasa or skrl packages.

Usage:
    from env.custom_turn_on_microwave import MyTurnOnMicrowave
    
    env = MyTurnOnMicrowave(
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

from robocasa.environments.kitchen.single_stage.kitchen_microwave import TurnOnMicrowave
import numpy as np


class MyTurnOnMicrowave(TurnOnMicrowave):
    """
    TurnOnMicrowave environment with modified reward function.
    
    This class inherits from the original TurnOnMicrowave and overrides
    the reward() method to implement a custom reward function.
    
    The task involves:
    1. Moving the gripper close to the microwave start button
    2. Pressing the start button to turn on the microwave
    3. Moving the gripper away from the button after pressing
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the custom environment.
        All arguments are passed to the parent TurnOnMicrowave class.
        """
        super().__init__(*args, **kwargs)
        
        # Reward weights
        self.W_REACH_BUTTON = 1.0      # Weight for reaching the button
        self.W_PRESS_BUTTON = 10.0     # Weight for pressing the button (turning on)
        self.W_RETREAT = 1.0           # Weight for moving gripper away after pressing
        self.W_SUCCESS = 5.0           # Weight for task completion
        self.W_ACTION_L2 = 0.01        # Penalty for large actions (smoothness)
        
        # Shaping scales
        self.ALPHA_REACH = 3.0         # Exponential decay rate for reach reward
        self.RETREAT_DISTANCE = 0.15   # Target distance to retreat from button (meters)
        
    def reward(self, action=None):
        """
        Custom reward function for the turn on microwave task.
        
        The reward is structured as follows:
        1. Reach reward: Encourage gripper to move close to the start button
        2. Press reward: Reward for successfully turning on the microwave
        3. Retreat reward: Encourage gripper to move away after pressing (only if turned on)
        4. Success reward: Bonus for completing the task (turned on + gripper far)
        5. Action penalty: Small penalty for large actions to encourage smooth movements
        
        Args:
            action (np.ndarray, optional): The action to compute the reward for. Defaults to None.
            
        Returns:
            float: Reward for the turn on microwave task
        """
        reward = 0.0
        
        # Get gripper position
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        
        # Get microwave state
        microwave_state = self.microwave.get_state()
        turned_on = microwave_state["turned_on"]
        
        # Get button position (start button)
        button_id = self.sim.model.geom_name2id(
            "{}start_button".format(self.microwave.naming_prefix)
        )
        button_pos = self.sim.data.geom_xpos[button_id]
        
        # Calculate distance from gripper to button
        dist_gripper_button = np.linalg.norm(gripper_site_pos - button_pos)
        
        # Check if gripper is far from button
        gripper_button_far = self.microwave.gripper_button_far(self, button="start_button")
        
        # 1. Reach Button: Encourage gripper to move close to the start button
        # Use exponential decay to provide smooth gradient
        r_reach = np.exp(-self.ALPHA_REACH * dist_gripper_button)
        reward += self.W_REACH_BUTTON * r_reach
        
        # 2. Press Button: Reward for successfully turning on the microwave
        if turned_on:
            reward += self.W_PRESS_BUTTON
            
            # 3. Retreat: After turning on, encourage gripper to move away
            # This helps the robot learn to complete the task properly
            if gripper_button_far:
                reward += self.W_RETREAT
            else:
                # Provide shaped reward for moving away
                # Encourage distance up to RETREAT_DISTANCE
                retreat_progress = min(dist_gripper_button / self.RETREAT_DISTANCE, 1.0)
                reward += self.W_RETREAT * retreat_progress
        
        # 4. Success: Full task completion (microwave on + gripper far from button)
        if self._check_success():
            reward += self.W_SUCCESS
        
        # 5. Action Penalty: Penalize large actions to encourage smooth movements
        if action is not None:
            action_penalty = self.W_ACTION_L2 * np.sum(np.square(action))
            reward -= action_penalty
        
        return float(reward)
    
    # ============================================================================
    # ALTERNATIVE REWARD FUNCTIONS
    # ============================================================================
    # Below are some example alternative reward functions you can experiment with.
    # To use one, rename it to 'reward' and rename the current 'reward' to something else.
    
    def reward_sparse(self, action=None):
        """
        Sparse reward: only reward on task completion.
        This is much harder to learn but avoids reward shaping bias.
        
        Returns:
            float: 10.0 if task is successful, 0.0 otherwise
        """
        if self._check_success():
            return 10.0
        return 0.0
    
    def reward_simple_dense(self, action=None):
        """
        Simple dense reward with basic distance-based shaping.
        
        Returns:
            float: Reward based on distance to button and task completion
        """
        reward = 0.0
        
        # Get positions
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        button_id = self.sim.model.geom_name2id(
            "{}start_button".format(self.microwave.naming_prefix)
        )
        button_pos = self.sim.data.geom_xpos[button_id]
        
        # Distance to button
        dist_gripper_button = np.linalg.norm(gripper_site_pos - button_pos)
        
        # Encourage reaching button with tanh shaping
        r_reach = 1 - np.tanh(5.0 * dist_gripper_button)
        reward += r_reach
        
        # Reward for turning on
        if self.microwave.get_state()["turned_on"]:
            reward += 5.0
        
        # Success bonus
        if self._check_success():
            reward += 2.0
        
        return reward
    
    def reward_staged(self, action=None):
        """
        Staged reward function that provides different rewards for different stages.
        Stage 1: Reach the button
        Stage 2: Press the button (turn on)
        Stage 3: Retreat from button
        
        Returns:
            float: Reward based on current stage of task completion
        """
        reward = 0.0
        
        # Get positions and state
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        button_id = self.sim.model.geom_name2id(
            "{}start_button".format(self.microwave.naming_prefix)
        )
        button_pos = self.sim.data.geom_xpos[button_id]
        dist_gripper_button = np.linalg.norm(gripper_site_pos - button_pos)
        turned_on = self.microwave.get_state()["turned_on"]
        gripper_button_far = self.microwave.gripper_button_far(self, button="start_button")
        
        # Stage 1: Reach button (always active until turned on)
        if not turned_on:
            # Strong shaping to reach button
            r_reach = np.exp(-5.0 * dist_gripper_button)
            reward += 2.0 * r_reach
            
            # Bonus for being very close to button
            if dist_gripper_button < 0.05:
                reward += 1.0
        
        # Stage 2: Press button
        if turned_on:
            reward += 10.0  # Large bonus for turning on
            
            # Stage 3: Retreat from button
            if not gripper_button_far:
                # Encourage moving away
                r_retreat = min(dist_gripper_button / 0.15, 1.0)
                reward += 2.0 * r_retreat
            else:
                # Bonus for being far enough
                reward += 2.0
        
        # Final success
        if self._check_success():
            reward += 5.0
        
        return reward
    
    def reward_with_velocity_penalty(self, action=None):
        """
        Reward function that penalizes high velocities to encourage careful movements.
        
        Returns:
            float: Reward with velocity penalty
        """
        reward = 0.0
        
        # Get positions
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        button_id = self.sim.model.geom_name2id(
            "{}start_button".format(self.microwave.naming_prefix)
        )
        button_pos = self.sim.data.geom_xpos[button_id]
        dist_gripper_button = np.linalg.norm(gripper_site_pos - button_pos)
        
        # Basic reach reward
        r_reach = np.exp(-3.0 * dist_gripper_button)
        reward += r_reach
        
        # Press reward
        turned_on = self.microwave.get_state()["turned_on"]
        if turned_on:
            reward += 10.0
            
            # Retreat reward
            gripper_button_far = self.microwave.gripper_button_far(self, button="start_button")
            if gripper_button_far:
                reward += 2.0
        
        # Success reward
        if self._check_success():
            reward += 5.0
        
        # Velocity penalty
        if action is not None:
            # Penalize large actions (which correspond to high velocities)
            velocity_penalty = 0.02 * np.sum(np.square(action))
            reward -= velocity_penalty
        
        # Additional penalty for very large single-step actions
        if action is not None and np.max(np.abs(action)) > 0.5:
            reward -= 0.5
        
        return reward


# Example of how to register this environment with robosuite if needed
def register_custom_env():
    """
    Register the custom environment with robosuite.
    This allows you to use robosuite.make("MyTurnOnMicrowave", ...)
    """
    import robosuite
    from robosuite.environments.base import register_env
    
    register_env(MyTurnOnMicrowave)
