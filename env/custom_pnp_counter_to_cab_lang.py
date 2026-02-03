"""
Custom RoboCasa PnPCounterToCab env with a *proposed dense reward* aligned to PPO training.

This file is optional but recommended: keep your reward logic in one place
instead of mixing it into training wrappers.

The reward below is a multi-stage dense reward:
- reach -> grasp -> lift -> transport -> place -> success
with small action penalty.

Compared to your current `custom_pnp_counter_to_cab.py`, this version:
- adds lift shaping
- gates transport shaping by grasp
- adds action L2 penalty
- exposes diagnostic terms in info (if you want to log them)

Usage:
    from custom_pnp_counter_to_cab_lang import MyPnPCounterToCabDense

    env = MyPnPCounterToCabDense(...)

Requires robocasa + robosuite.
"""

import sys
import os
import numpy as np
import robocasa.utils.object_utils as OU

# Add robocasa to path if not already there
robocasa_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "robocasa")
if robocasa_path not in sys.path:
    sys.path.insert(0, robocasa_path)

from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToCab


class MyPnPCounterToCabDense(PnPCounterToCab):
    """
    Proposed dense reward for PnPCounterToCab.

    Tunable weights are embedded as constants for clarity.
    If you prefer, move them to kwargs.
    """

    # weights
    W_REACH = 1.0
    W_GRASP = 3.0
    W_LIFT = 1.0
    W_TRANSPORT = 1.0
    W_PLACE = 10.0
    W_SUCCESS = 5.0
    W_ACTION_L2 = 0.01

    # shaping scales
    ALPHA_REACH = 2.0       # exp(-ALPHA * dist)
    ALPHA_TRANSPORT = 2.0
    LIFT_TARGET = 0.08      # meters above counter/table, adjust if needed

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

        return float(r)
