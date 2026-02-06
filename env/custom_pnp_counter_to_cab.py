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
    PnPCounterToCab environment with modified reward function.
    
    This class inherits from the original PnPCounterToCab and overrides
    the reward() method to implement a custom reward function.
    
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the custom environment.
        All arguments are passed to the parent PnPCounterToCab class.
        
        This environment fixes the kitchen layout to a single configuration
        while allowing object positions to vary.
        """
        # Fix the kitchen layout to layout 0, style 0 (you can change these values)
        # This prevents the kitchen configuration from changing between episodes
        if 'layout_ids' not in kwargs:
            kwargs['layout_ids'] = [1]  # Use layout 1
        if 'style_ids' not in kwargs:
            kwargs['style_ids'] = [1]   # Use style 1
            
        # Capture the seed if provided
        self.custom_seed = kwargs.get('seed', 0)
        
        super().__init__(*args, **kwargs)
        
        # weights
        self.W_REACH = 1.0
        self.W_GRASP = 5.0
        self.W_LIFT = 1.0
        self.W_TRANSPORT = 1.0
        self.W_PLACE = 10.0
        self.W_SUCCESS = 5.0
        self.W_ACTION_L2 = 0.01

        # shaping scales
        self.ALPHA_REACH = 2.0       # exp(-ALPHA * dist)
        self.ALPHA_TRANSPORT = 2.0
        self.LIFT_TARGET = 0.08      # meters above counter/table, adjust if needed
    
    def _get_obj_cfgs(self):
        """
        Override to set specific objects:
        - Sample object (obj): always apple_1
        - Distractor objects: always bowl_1
        """
        import os
        import robocasa
        
        cfgs = []
        
        # Get the base path for robocasa objects
        base_path = os.path.join(robocasa.models.assets_root, "objects", "objaverse")
        # print("base_path", base_path)
        # Sample object: always apple_1 (using full path to model.xml)
        apple_1_path = os.path.join(base_path, "apple", "apple_1", "model.xml")
        # print("apple_1_path", apple_1_path)
        cfgs.append(
            dict(
                name="obj",
                obj_groups=apple_1_path,  # Force apple_1 as the sample object
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.60, 0.30),
                    pos=(0.0, -1.0),
                    offset=(0.0, 0.10),
                ),
            )
        )

        # Distractor on counter: always bowl_1 (using full path to model.xml)
        bowl_1_path = os.path.join(base_path, "bowl", "bowl_1", "model.xml")
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups=bowl_1_path,  # Force bowl_1 as distractor
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )

        return cfgs
        
    def _get_placement_initializer(self, cfg_list, z_offset=0.01):
        """
        Override to enforce deterministic placement for fixtures (appliances),
        while allowing random placement for objects.
        """
        sampler = super()._get_placement_initializer(cfg_list, z_offset)
        
        # Check if this sampler is for fixtures (appliances)
        # Fixture configs usually have type="fixture"
        is_fixture_placement = False
        if cfg_list and len(cfg_list) > 0:
            if cfg_list[0].get("type") == "fixture":
                is_fixture_placement = True
        
        if is_fixture_placement:
            # Use the environment seed for fixture placement to ensure deterministic behavior
            # appliances will stay in place throughout the run (assuming constant seed for env)
            # but will change if you change the training run seed.
            
            # Retrieve the seed captured in __init__
            seed_val = getattr(self, "custom_seed", 0)
            if seed_val is None:
                seed_val = 0
                
            fixed_rng = np.random.default_rng(seed=seed_val)
            
            # Set the RNG for the main sampler
            sampler.rng = fixed_rng
            
            # Set the RNG for all sub-samplers
            if hasattr(sampler, "samplers"):
                for sub_sampler in sampler.samplers.values():
                    sub_sampler.rng = fixed_rng
                    
        return sampler
        
    def reward(self, action=None):
        r = 0.0

        # Positions
        obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]]
        ee_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        cab_pos = np.array(self.cab.pos, dtype=np.float32)

        # Distances
        dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))
        dist_obj_cab = float(np.linalg.norm(obj_pos - cab_pos))

        # Stage signals
        is_grasped = bool(self.check_contact(self.objects["obj"], self.robots[0].gripper))
        is_inside = bool(OU.obj_inside_of(self, "obj", self.cab))
        is_success = bool(self._check_success())

        # 1) Reach object (always on)
        r_reach = np.exp(-self.ALPHA_REACH * dist_ee_obj)
        r += self.W_REACH * r_reach

        # 2) Grasp bonus (sparse but frequent once learned)
        if is_grasped:
            r += self.W_GRASP

            # 3) Lift shaping (only after grasp)
            # We encourage lifting a bit to reduce dragging collisions.
            # Estimate counter height as initial z if available; here we use cab/counter logic indirectly.
            lift = max(0.0, float(obj_pos[2]) - float(self.table_offset[2]))  # rough baseline
            r_lift = np.clip(lift / self.LIFT_TARGET, 0.0, 1.0)
            r += self.W_LIFT * r_lift

            # 4) Transport to cabinet (only after grasp)
            r_transport = np.exp(-self.ALPHA_TRANSPORT * dist_obj_cab)
            r += self.W_TRANSPORT * r_transport

        # 5) Place bonus (object inside cabinet volume)
        if is_inside:
            r += self.W_PLACE

        # 6) Success bonus
        if is_success:
            r += self.W_SUCCESS

        # Penalty: large actions (smoothness)
        if action is not None:
            r -= self.W_ACTION_L2 * float(np.sum(np.square(action)))

        # Penalize touching distractor objects
        if self.check_contact(self.objects["distr_counter"], self.robots[0].gripper):
            r -= 0.5

        return float(r)
    
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
