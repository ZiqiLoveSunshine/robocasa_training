"""
Real-time visualization script for RoboCasa environment.

This script creates two windows:
1. Simulator window - shows the main 3D environment view
2. Camera views window - shows all 6 camera perspectives in a 2x3 grid that update in real-time
   - robot0_agentview_center, left, right
   - robot0_frontview, robotview
   - robot0_eye_in_hand

You can teleoperate the robot using keyboard controls, and all views update simultaneously.
"""

import sys
import os

# Use glfw for better compatibility with both onscreen and offscreen rendering
# This allows the simulator window and camera views to work together
os.environ["MUJOCO_GL"] = "glfw"

# Add local robocasa to path
robocasa_path = os.path.join(os.path.dirname(__file__), 'robocasa')
sys.path.insert(0, robocasa_path)

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.devices import Keyboard
import robocasa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import threading
import time


class RoboCasaVisualizer:
    """
    Real-time visualizer for RoboCasa environments with multiple camera views.
    """
    
    def __init__(self, env_name="PnPCounterToCab", robot="PandaOmron", 
                 camera_names=None, camera_height=256, camera_width=256, seed=None):
        """
        Initialize the visualizer.
        
        Args:
            env_name: Name of the RoboCasa environment
            robot: Robot model to use
            camera_names: List of camera names to visualize
            camera_height: Height of camera images
            camera_width: Width of camera images
            seed: Random seed for environment (None for random)
        """
        self.env_name = env_name
        self.robot = robot
        self.seed = seed
        
        # Default camera names if not provided
        if camera_names is None:
            # Include all 6 available cameras for comprehensive view
            self.camera_names = [
                "robot0_agentview_center",
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_frontview",
                "robot0_robotview",
                "robot0_eye_in_hand"
            ]
        else:
            self.camera_names = camera_names
        
        self.camera_height = camera_height
        self.camera_width = camera_width
        
        # Initialize environment
        self.env = None
        self.obs = None
        self.running = True
        self.action = None
        
        # For keyboard control
        self.device = None
        
    def create_environment(self):
        """Create the RoboCasa environment with rendering and cameras enabled."""
        print(f"Creating {self.env_name} environment...")
        
        # Setup controller
        controller_config = load_composite_controller_config(
            controller=None,
            robot=self.robot,
        )
        
        # Create environment with both onscreen and offscreen rendering
        env_kwargs = dict(
            env_name=self.env_name,
            robots=self.robot,
            controller_configs=controller_config,
            has_renderer=True,  # For simulator window
            has_offscreen_renderer=True,  # For camera views
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_names=self.camera_names,
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
            control_freq=20,
            reward_shaping=True,
            render_camera="robot0_frontview",  # Main simulator view
            seed=self.seed,  # Random seed for reproducibility
        )
        
        self.env = robosuite.make(**env_kwargs)
        
        # Reset environment
        print("Resetting environment...")
        if self.seed is not None:
            print(f"Using seed: {self.seed}")
        self.obs = self.env.reset()
        
        # Print task information
        ep_meta = self.env.get_ep_meta()
        task_description = ep_meta.get("lang", "Task")
        print(f"\nTask: {task_description}")
        print(f"Available cameras: {self.camera_names}")
        print(f"Action space: {self.env.action_spec[0].shape}")
        
    def setup_keyboard_control(self):
        """Setup keyboard device for teleoperation."""
        print("\nSetting up keyboard control...")
        print("Use keyboard to control the robot:")
        print("  - Arrow keys: Move end-effector")
        print("  - W/S: Move up/down")
        print("  - A/D: Rotate")
        print("  - Space: Toggle gripper")
        print("  - ESC: Quit")
        
        self.device = Keyboard(env=self.env, pos_sensitivity=1.0, rot_sensitivity=1.0)
        self.device.start_control()
        
    def get_camera_images(self):
        """Extract camera images from current observation."""
        images = []
        for cam_name in self.camera_names:
            img_key = f"{cam_name}_image"
            if img_key in self.obs:
                img = self.obs[img_key]
                # Convert from (H, W, C) to RGB if needed
                if img.shape[-1] == 3:
                    images.append(img)
                else:
                    # Handle grayscale or other formats
                    images.append(np.stack([img]*3, axis=-1))
            else:
                # Create placeholder if camera not available
                placeholder = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
                images.append(placeholder)
        
        return images
    
    def run_simulator_window(self):
        """Run the main simulator window in a separate thread."""
        print("\nStarting simulator window...")
        
        while self.running:
            # Render the simulator view
            self.env.render()
            
            # Small delay to control frame rate
            time.sleep(0.01)
    
    def run_camera_visualization(self):
        """Create and run the camera views window with live updates."""
        print("Starting camera views window...")
        
        # Create figure with subplots for each camera
        num_cameras = len(self.camera_names)
        
        # Use a 2x3 grid for 6 cameras, or adjust based on number
        if num_cameras <= 3:
            rows, cols = 1, num_cameras
        elif num_cameras <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 2, (num_cameras + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        # Flatten axes array for easier iteration
        if num_cameras == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Initialize image displays
        img_displays = []
        for idx, (ax, cam_name) in enumerate(zip(axes, self.camera_names)):
            ax.set_title(cam_name.replace("robot0_", "").replace("_", " ").title(), 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Get initial image
            images = self.get_camera_images()
            img_display = ax.imshow(images[idx])
            img_displays.append(img_display)
        
        # Hide unused subplots if we have more axes than cameras
        for idx in range(num_cameras, len(axes)):
            axes[idx].axis('off')
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        def update_frame(frame):
            """Update function for animation."""
            if not self.running:
                return img_displays
            
            # Get keyboard input and convert to action
            if self.device is not None:
                state = self.device.get_controller_state()
                
                # Get arm actions (7D: 3 pos, 3 rot, 1 gripper)
                # Ensure all arrays are properly flattened
                dpos = np.array(state.get("dpos", np.zeros(3))).flatten()
                rotation = np.array(state.get("rotation", np.zeros(3))).flatten()
                grasp = np.array([state.get("grasp", 0)]).flatten()
                
                arm_action = np.concatenate([dpos, rotation, grasp])
                
                # Get base actions (3D: x, y, rotation) - use 'b' key to toggle base mode
                # For now, we'll set base to zero (stationary) unless in base mode
                base_action = state.get("base_action", np.zeros(3))
                
                # Combine: [arm_actions (7D), base_actions (3D), padding (2D)] = 12D
                # The exact structure depends on the robot, but typically:
                # [arm_dpos(3), arm_rot(3), gripper(1), base_x, base_y, base_rot, ...]
                action = np.concatenate([arm_action, base_action])
                
                # Pad to match action dimension if needed
                if len(action) < self.env.action_dim:
                    action = np.concatenate([action, np.zeros(self.env.action_dim - len(action))])
                elif len(action) > self.env.action_dim:
                    action = action[:self.env.action_dim]
            else:
                # Random action if no keyboard
                action = np.random.uniform(
                    self.env.action_spec[0], 
                    self.env.action_spec[1]
                )
            
            # Step environment
            self.obs, reward, done, info = self.env.step(action)
            
            # Update camera images
            images = self.get_camera_images()
            for img_display, img in zip(img_displays, images):
                img_display.set_array(img)
            
            # Print reward info occasionally
            if frame % 20 == 0:
                print(f"Step {frame}, Reward: {reward:.4f}")
            
            return img_displays
        
        # Create animation
        anim = FuncAnimation(
            fig, 
            update_frame, 
            interval=50,  # 50ms = 20 FPS
            blit=True,
            cache_frame_data=False
        )
        
        # Show the plot
        plt.show()
        
        # When window is closed, stop everything
        self.running = False
    
    def run(self, use_keyboard=True):
        """
        Run the complete visualization with both windows.
        
        Args:
            use_keyboard: If True, enable keyboard teleoperation
        """
        # Create environment
        self.create_environment()
        
        # Setup keyboard control if requested
        if use_keyboard:
            self.setup_keyboard_control()
        
        # Start simulator window in separate thread
        simulator_thread = threading.Thread(target=self.run_simulator_window)
        simulator_thread.daemon = True
        simulator_thread.start()
        
        # Run camera visualization in main thread (matplotlib needs main thread)
        try:
            self.run_camera_visualization()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        self.running = False
        
        if self.device is not None:
            print("Stopping keyboard control...")
            # The device will be cleaned up when env closes
        
        if self.env is not None:
            print("Closing environment...")
            self.env.close()
        
        print("Done!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time visualization of RoboCasa environment with multiple camera views"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        default="PnPCounterToCab",
        help="Environment name (default: PnPCounterToCab)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="PandaOmron",
        help="Robot model (default: PandaOmron)"
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="Camera names to visualize (default: all 6 cameras - agentview_center/left/right, frontview, robotview, eye_in_hand)"
    )
    parser.add_argument(
        "--camera-size",
        type=int,
        default=256,
        help="Camera image size (default: 256)"
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable keyboard control (use random actions)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for environment (default: None for random)"
    )
    
    args = parser.parse_args()
    
    # Create and run visualizer
    visualizer = RoboCasaVisualizer(
        env_name=args.env,
        robot=args.robot,
        camera_names=args.cameras,
        camera_height=args.camera_size,
        camera_width=args.camera_size,
        seed=args.seed
    )
    
    visualizer.run(use_keyboard=not args.no_keyboard)


if __name__ == "__main__":
    main()
