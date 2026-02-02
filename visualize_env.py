import sys
import os

# FORCE EGL for offscreen rendering
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Add local robocasa to path
robocasa_path = os.path.join(os.path.dirname(__file__), 'robocasa')
sys.path.insert(0, robocasa_path)

import robosuite
from robosuite.controllers import load_composite_controller_config
import robocasa
import numpy as np
from PIL import Image
import argparse

def visualize_pnp_counter_to_cab(save_path="pnp_counter_to_cab_viz.png", num_views=4):
    """
    Visualize the PnPCounterToCab environment from multiple camera angles
    """
    # Setup controller
    robots = "PandaOmron"
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots,
    )
    
    # Create environment with multiple cameras
    env_kwargs = dict(
        env_name="PnPCounterToCab",
        robots=robots,
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_center", "robot0_eye_in_hand", "robot0_frontview"],
        camera_heights=512,
        camera_widths=512,
        control_freq=20,
        seed=1,
        reward_shaping=True,
    )

    print("Creating PnPCounterToCab environment...")
    env = robosuite.make(**env_kwargs)
    
    # Reset environment to get initial state
    print("Resetting environment...")
    obs = env.reset()
    
    # Get episode metadata
    ep_meta = env.get_ep_meta()
    task_description = ep_meta.get("lang", "Pick and place task")
    print(f"\nTask Description: {task_description}")
    
    # Print object information
    print(f"\nObjects in scene:")
    for obj_name, obj in env.objects.items():
        print(f"  - {obj_name}: {obj.name}")
    
    # Capture images from different views
    images = []
    camera_names = ["robot0_agentview_center_image", "robot0_eye_in_hand_image", "robot0_frontview_image"]
    
    print("\nCapturing camera views...")
    for cam_name in camera_names:
        if cam_name in obs:
            img = obs[cam_name]
            images.append(img)
            print(f"  - Captured {cam_name}: {img.shape}")
    
    # Also take a few random actions and capture more views
    print("\nTaking random actions to show different states...")
    for i in range(3):
        action = np.random.uniform(env.action_spec[0], env.action_spec[1])
        obs, reward, done, info = env.step(action)
        if "robot0_agentview_center_image" in obs:
            images.append(obs["robot0_agentview_center_image"])
    
    # Create a grid of images
    print(f"\nCreating visualization grid with {len(images)} images...")
    if len(images) >= 4:
        # Create 2x3 grid
        rows = 2
        cols = 3
        grid_images = images[:6]
    else:
        rows = 1
        cols = len(images)
        grid_images = images
    
    # Ensure all images have same size
    h, w = grid_images[0].shape[:2]
    
    # Create combined image
    combined_height = h * rows
    combined_width = w * cols
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    for idx, img in enumerate(grid_images):
        row = idx // cols
        col = idx % cols
        combined_image[row*h:(row+1)*h, col*w:(col+1)*w] = img
    
    # Save image
    pil_image = Image.fromarray(combined_image)
    pil_image.save(save_path)
    print(f"\nVisualization saved to: {save_path}")
    
    # Print environment details
    print("\n" + "="*60)
    print("ENVIRONMENT DETAILS")
    print("="*60)
    print(f"Task: PnPCounterToCab")
    print(f"Description: {task_description}")
    print(f"Robot: {robots}")
    print(f"Action Space: {env.action_spec[0].shape}")
    print(f"Action Range: [{env.action_spec[0].min():.2f}, {env.action_spec[1].max():.2f}]")
    print(f"Number of Objects: {len(env.objects)}")
    print("="*60)
    
    env.close()
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="pnp_counter_to_cab_viz.png", 
                       help="Output path for visualization")
    args = parser.parse_args()
    
    visualize_pnp_counter_to_cab(save_path=args.output)
