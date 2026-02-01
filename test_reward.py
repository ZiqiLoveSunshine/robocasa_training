import robocasa
from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToCab
import numpy as np

def test_reward():
    env = PnPCounterToCab(robots="Panda", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, render_camera="robot0_robotview")
    env.reset()
    
    print("Initial State:")
    r = env.reward()
    print(f"Reward: {r}")
    
    # Get object position
    obj_pos = env.sim.data.body_xpos[env.obj_body_id["obj"]]
    print(f"Object Pos: {obj_pos}")
    
    # Get gripper position
    gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
    print(f"Gripper Pos: {gripper_site_pos}")

    dist_reach = np.linalg.norm(gripper_site_pos - obj_pos)
    print(f"Distance Reach: {dist_reach}")
    
    expected_reach_reward = 1 - np.tanh(5.0 * dist_reach)
    print(f"Expected Reach Objekt Reward: {expected_reach_reward}")
    
    # Check if reward matches parts
    dist_place = np.linalg.norm(obj_pos - env.cab.pos)
    expected_place_reward = 1 - np.tanh(5.0 * dist_place)
    print(f"Distance to Cab: {dist_place}")
    print(f"Expected Reach Cabinet Reward: {expected_place_reward}")
    
    # Initial state: not grasped, not placed, not success
    # Expect: Reach Object + Reach Cabinet
    total_initial = expected_reach_reward + expected_place_reward
    print(f"Total Expected (Initial): {total_initial}")
    assert np.isclose(r, total_initial), f"Initial reward mismatch: {r} != {total_initial}"
    
    
    # Test Grasp
    # Hard to simulate grasp without running simulation, but we can assume if check_contact works it adds 5.0
    # Let's skip hard simulation of grasp in this unit test unless we mock it, 
    # but we can verify the other static states.
    
    # Test Place (Object inside cabinet)
    print("\nTeleporting object to cabinet to test Place + Reach Cabinet + Success...")
    cab_pos = env.cab.pos
    # Offset slightly to be validly strictly inside
    target_pos = cab_pos + np.array([0, 0, 0.1]) 
    
    env.sim.data.set_joint_qpos(
        env.objects["obj"].joints[0],
        np.concatenate([target_pos, np.array([1, 0, 0, 0])])
    )
    env.sim.forward()
    
    new_obj_pos = env.sim.data.body_xpos[env.obj_body_id["obj"]]
    new_gripper_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
    
    dist_reach_new = np.linalg.norm(new_gripper_pos - new_obj_pos)
    r_reach_obj_new = 1 - np.tanh(5.0 * dist_reach_new)
    
    dist_place_new = np.linalg.norm(new_obj_pos - env.cab.pos)
    r_reach_cab_new = 1 - np.tanh(5.0 * dist_place_new)
    
    is_placed = True # We put it in
    r_grouped_place = 5.0 if is_placed else 0.0
    
    # Success requires gripper far.
    # We haven't moved gripper, so it might be far.
    is_success = env._check_success()
    r_success = 2.0 if is_success else 0.0
    
    # We are NOT grasping it (teleported object away from gripper)
    is_grasped = False
    r_grasp = 5.0 if is_grasped else 0.0
    
    expected_result_teleport = r_reach_obj_new + r_grasp + r_reach_cab_new + r_grouped_place + r_success
    
    current_reward = env.reward()
    print(f"Current Reward (In Cab): {current_reward}")
    print(f"Expected components: ReachObj={r_reach_obj_new:.3f}, Grasp={r_grasp}, ReachCab={r_reach_cab_new:.3f}, Place={r_grouped_place}, Success={r_success}")
    print(f"Total Expected: {expected_result_teleport}")
    
    # Allow small float error
    assert np.isclose(current_reward, expected_result_teleport, atol=1e-4), f"In-Cab reward mismatch: {current_reward} != {expected_result_teleport}"

    if is_success:
        print("Success condition triggered correctly.")
    else:
        print("Success condition NOT triggered (gripper might be too close).")

    print("Reward verification passed!")

if __name__ == "__main__":
    test_reward()
