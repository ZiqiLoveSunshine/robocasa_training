"""
PID Controller for RoboCasa Robot Home Position Control

This script demonstrates a PID controller that keeps the robot at its home position.
The controller computes actions to minimize the error between current and desired positions.
"""

import sys
import os

# Use glfw for better compatibility with both onscreen and offscreen rendering
os.environ["MUJOCO_GL"] = "glfw"

# Add local robocasa to path
robocasa_path = os.path.join(os.path.dirname(__file__), 'robocasa')
sys.path.insert(0, robocasa_path)

import robosuite
from robosuite.controllers import load_composite_controller_config
import robocasa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import threading
import time


class PIDController:
    """
    PID Controller for position control.
    
    The PID controller computes a control signal based on:
    - Proportional term (P): Current error
    - Integral term (I): Accumulated error over time
    - Derivative term (D): Rate of change of error
    """
    
    def __init__(self, kp, ki, kd, output_limits=None):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Tuple of (min, max) for output clamping
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        # State variables
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None
        
    def reset(self):
        """Reset the controller state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None
        
    def compute(self, error, dt=None):
        """
        Compute PID output.
        
        Args:
            error: Current error (setpoint - measurement)
            dt: Time step (optional, will use time.time() if None)
            
        Returns:
            Control output
        """
        # Handle time step
        if dt is None:
            current_time = time.time()
            if self.previous_time is None:
                dt = 0.0
            else:
                dt = current_time - self.previous_time
            self.previous_time = current_time
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative
        
        # Compute total output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_limits is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
            
            # Anti-windup: prevent integral from growing if output is saturated
            if output == self.output_limits[0] or output == self.output_limits[1]:
                self.integral -= error * dt
        
        # Update state
        self.previous_error = error
        
        return output


class VectorPIDController:
    """PID controller for multi-dimensional vectors (e.g., 3D position)."""
    
    def __init__(self, kp, ki, kd, dim, output_limits=None):
        """
        Initialize vector PID controller.
        
        Args:
            kp: Proportional gain (scalar or array)
            ki: Integral gain (scalar or array)
            kd: Derivative gain (scalar or array)
            dim: Dimension of the control vector
            output_limits: Tuple of (min, max) for output clamping
        """
        # Convert gains to arrays if they're scalars
        self.kp = np.ones(dim) * kp if np.isscalar(kp) else np.array(kp)
        self.ki = np.ones(dim) * ki if np.isscalar(ki) else np.array(ki)
        self.kd = np.ones(dim) * kd if np.isscalar(kd) else np.array(kd)
        
        self.dim = dim
        self.output_limits = output_limits
        
        # State variables
        self.integral = np.zeros(dim)
        self.previous_error = np.zeros(dim)
        self.previous_time = None
        
    def reset(self):
        """Reset the controller state."""
        self.integral = np.zeros(self.dim)
        self.previous_error = np.zeros(self.dim)
        self.previous_time = None
        
    def compute(self, error, dt=None):
        """
        Compute PID output for vector error.
        
        Args:
            error: Current error vector (setpoint - measurement)
            dt: Time step (optional)
            
        Returns:
            Control output vector
        """
        error = np.array(error)
        
        # Handle time step
        if dt is None:
            current_time = time.time()
            if self.previous_time is None:
                dt = 0.02  # Default 50Hz
            else:
                dt = current_time - self.previous_time
            self.previous_time = current_time
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = np.zeros(self.dim)
        d_term = self.kd * derivative
        
        # Compute total output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_limits is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update state
        self.previous_error = error.copy()
        
        return output


class RoboCasaPIDHomePosition:
    """
    RoboCasa environment with PID control to maintain home position.
    """
    
    def __init__(self, env_name="PnPCounterToCab", robot="PandaOmron",
                 camera_names=None, camera_height=256, camera_width=256,
                 kp_pos=1.0, ki_pos=0.0, kd_pos=0.1,
                 kp_ori=0.5, ki_ori=0.0, kd_ori=0.05, seed=None):
        """
        Initialize the PID home position controller.
        
        Args:
            env_name: Name of the RoboCasa environment
            robot: Robot model to use
            camera_names: List of camera names to visualize
            camera_height: Height of camera images
            camera_width: Width of camera images
            kp_pos: Proportional gain for position
            ki_pos: Integral gain for position
            kd_pos: Derivative gain for position
            kp_ori: Proportional gain for orientation
            ki_ori: Integral gain for orientation
            kd_ori: Derivative gain for orientation
            seed: Random seed for environment (None for random)
        """
        self.env_name = env_name
        self.robot = robot
        self.seed = seed
        
        # Default camera names if not provided
        if camera_names is None:
            self.camera_names = [
                "robot0_agentview_center",
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ]
        else:
            self.camera_names = camera_names
        
        self.camera_height = camera_height
        self.camera_width = camera_width
        
        # Initialize environment
        self.env = None
        self.obs = None
        self.running = True
        
        # Home position (will be set after environment reset)
        self.home_eef_pos = None
        self.home_eef_quat = None
        
        # PID controllers
        # Position controller (3D)
        self.pid_position = VectorPIDController(
            kp=kp_pos, ki=ki_pos, kd=kd_pos, dim=3,
            output_limits=(-0.05, 0.05)  # Limit position commands
        )
        
        # Orientation controller (3D - axis-angle representation)
        self.pid_orientation = VectorPIDController(
            kp=kp_ori, ki=ki_ori, kd=kd_ori, dim=3,
            output_limits=(-0.1, 0.1)  # Limit rotation commands
        )
        
        # Data logging
        self.position_errors = []
        self.orientation_errors = []
        self.timestamps = []
        self.start_time = None
        
    def create_environment(self):
        """Create the RoboCasa environment."""
        print(f"Creating {self.env_name} environment...")
        
        # Setup controller
        controller_config = load_composite_controller_config(
            controller=None,
            robot=self.robot,
        )
        
        # Create environment
        env_kwargs = dict(
            env_name=self.env_name,
            robots=self.robot,
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_names=self.camera_names,
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
            control_freq=20,
            reward_shaping=True,
            render_camera="robot0_frontview",
            seed=self.seed,
        )
        
        self.env = robosuite.make(**env_kwargs)
        
        # Reset environment with seed if provided
        print("Resetting environment...")
        if self.seed is not None:
            print(f"Using seed: {self.seed}")
            # Set numpy random seed before reset
            np.random.seed(self.seed)
        self.obs = self.env.reset()
        
        # Store home position (current position after reset)
        self.home_eef_pos = self.obs["robot0_eef_pos"].copy()
        self.home_eef_quat = self.obs["robot0_eef_quat"].copy()
        
        print(f"\nHome position set:")
        print(f"  Position: {self.home_eef_pos}")
        print(f"  Quaternion: {self.home_eef_quat}")
        
        # Print task information
        ep_meta = self.env.get_ep_meta()
        task_description = ep_meta.get("lang", "Task")
        print(f"\nTask: {task_description}")
        print(f"Available cameras: {self.camera_names}")
        print(f"Action space: {self.env.action_spec[0].shape}")
        
        self.start_time = time.time()
        
    def quaternion_to_axis_angle(self, quat):
        """
        Convert quaternion to axis-angle representation.
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            Axis-angle vector (3D)
        """
        w, x, y, z = quat
        angle = 2 * np.arccos(np.clip(w, -1, 1))
        
        if angle < 1e-6:
            return np.zeros(3)
        
        s = np.sqrt(1 - w*w)
        if s < 1e-6:
            return np.array([x, y, z]) * angle
        
        axis = np.array([x, y, z]) / s
        return axis * angle
    
    def compute_orientation_error(self, current_quat, target_quat):
        """
        Compute orientation error between two quaternions.
        
        Args:
            current_quat: Current orientation quaternion
            target_quat: Target orientation quaternion
            
        Returns:
            Orientation error as axis-angle vector
        """
        # Compute relative quaternion: q_error = q_target * q_current^-1
        # Quaternion format: [w, x, y, z]
        w1, x1, y1, z1 = target_quat
        w2, x2, y2, z2 = current_quat
        
        # Conjugate of current quaternion
        w2_conj, x2_conj, y2_conj, z2_conj = w2, -x2, -y2, -z2
        
        # Multiply: q_error = q_target * q_current_conj
        w = w1*w2_conj - x1*x2_conj - y1*y2_conj - z1*z2_conj
        x = w1*x2_conj + x1*w2_conj + y1*z2_conj - z1*y2_conj
        y = w1*y2_conj - x1*z2_conj + y1*w2_conj + z1*x2_conj
        z = w1*z2_conj + x1*y2_conj - y1*x2_conj + z1*w2_conj
        
        q_error = np.array([w, x, y, z])
        
        # Convert to axis-angle
        return self.quaternion_to_axis_angle(q_error)
    
    def compute_pid_action(self):
        """
        Compute PID control action to return to home position.
        
        Returns:
            Action array for the environment
        """
        # Get current end-effector state
        current_pos = self.obs["robot0_eef_pos"]
        current_quat = self.obs["robot0_eef_quat"]
        
        # Compute position error
        pos_error = self.home_eef_pos - current_pos
        
        # Compute orientation error
        ori_error = self.compute_orientation_error(current_quat, self.home_eef_quat)
        
        # Compute PID outputs
        pos_action = self.pid_position.compute(pos_error)
        ori_action = self.pid_orientation.compute(ori_error)
        
        # Log errors
        self.position_errors.append(np.linalg.norm(pos_error))
        self.orientation_errors.append(np.linalg.norm(ori_error))
        self.timestamps.append(time.time() - self.start_time)
        
        # Construct full action: [pos(3), ori(3), gripper(1), base(3), padding(2)]
        # Keep gripper closed (0) and base stationary (0,0,0)
        action = np.concatenate([
            pos_action,
            ori_action,
            [0.0],  # Gripper
            np.zeros(3),  # Base
            np.zeros(2)  # Padding
        ])
        
        # Ensure action matches expected dimension
        if len(action) < self.env.action_dim:
            action = np.concatenate([action, np.zeros(self.env.action_dim - len(action))])
        elif len(action) > self.env.action_dim:
            action = action[:self.env.action_dim]
        
        return action
    
    def get_camera_images(self):
        """Extract camera images from current observation."""
        images = []
        for cam_name in self.camera_names:
            img_key = f"{cam_name}_image"
            if img_key in self.obs:
                img = self.obs[img_key]
                if img.shape[-1] == 3:
                    images.append(img)
                else:
                    images.append(np.stack([img]*3, axis=-1))
            else:
                placeholder = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
                images.append(placeholder)
        
        return images
    
    def run_simulator_window(self):
        """Run the main simulator window in a separate thread."""
        print("\nStarting simulator window...")
        
        while self.running:
            self.env.render()
            time.sleep(0.01)
    
    def run_camera_visualization(self):
        """Create and run the camera views window with live updates."""
        print("Starting camera views window with PID control...")
        
        # Create figure with subplots for each camera
        num_cameras = len(self.camera_names)
        fig, axes = plt.subplots(1, num_cameras, figsize=(6*num_cameras, 6))
        
        if num_cameras == 1:
            axes = [axes]
        
        # Initialize image displays
        img_displays = []
        for idx, (ax, cam_name) in enumerate(zip(axes, self.camera_names)):
            ax.set_title(cam_name.replace("robot0_", "").replace("_", " ").title(), 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            images = self.get_camera_images()
            img_display = ax.imshow(images[idx])
            img_displays.append(img_display)
        
        plt.tight_layout()
        
        def update_frame(frame):
            """Update function for animation."""
            if not self.running:
                return img_displays
            
            # Compute PID action
            action = self.compute_pid_action()
            
            # Step environment
            self.obs, reward, done, info = self.env.step(action)
            
            # Update camera images
            images = self.get_camera_images()
            for img_display, img in zip(img_displays, images):
                img_display.set_array(img)
            
            # Print status occasionally
            if frame % 20 == 0:
                pos_error = self.position_errors[-1] if self.position_errors else 0
                ori_error = self.orientation_errors[-1] if self.orientation_errors else 0
                print(f"Step {frame}, Pos Error: {pos_error:.6f}m, Ori Error: {ori_error:.6f}rad, Reward: {reward:.4f}")
            
            return img_displays
        
        # Create animation
        anim = FuncAnimation(
            fig, 
            update_frame, 
            interval=50,  # 50ms = 20 FPS
            blit=True,
            cache_frame_data=False
        )
        
        plt.show()
        self.running = False
    
    def run(self):
        """Run the complete visualization with PID control."""
        # Create environment
        self.create_environment()
        
        # Start simulator window in separate thread
        simulator_thread = threading.Thread(target=self.run_simulator_window)
        simulator_thread.daemon = True
        simulator_thread.start()
        
        # Run camera visualization in main thread
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
        
        if self.env is not None:
            print("Closing environment...")
            self.env.close()
        
        # Plot error history
        if len(self.timestamps) > 0:
            self.plot_errors()
        
        print("Done!")
    
    def plot_errors(self):
        """Plot position and orientation errors over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Position error
        ax1.plot(self.timestamps, self.position_errors, 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Position Error (m)', fontsize=12)
        ax1.set_title('Position Error Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Orientation error
        ax2.plot(self.timestamps, self.orientation_errors, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Orientation Error (rad)', fontsize=12)
        ax2.set_title('Orientation Error Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PID controller to maintain robot at home position"
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
        "--kp-pos",
        type=float,
        default=1.0,
        help="Proportional gain for position (default: 1.0)"
    )
    parser.add_argument(
        "--ki-pos",
        type=float,
        default=0.0,
        help="Integral gain for position (default: 0.0)"
    )
    parser.add_argument(
        "--kd-pos",
        type=float,
        default=0.1,
        help="Derivative gain for position (default: 0.1)"
    )
    parser.add_argument(
        "--kp-ori",
        type=float,
        default=0.5,
        help="Proportional gain for orientation (default: 0.5)"
    )
    parser.add_argument(
        "--ki-ori",
        type=float,
        default=0.0,
        help="Integral gain for orientation (default: 0.0)"
    )
    parser.add_argument(
        "--kd-ori",
        type=float,
        default=0.05,
        help="Derivative gain for orientation (default: 0.05)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for environment (default: None for random)"
    )
    
    args = parser.parse_args()
    
    # Create and run PID controller
    controller = RoboCasaPIDHomePosition(
        env_name=args.env,
        robot=args.robot,
        kp_pos=args.kp_pos,
        ki_pos=args.ki_pos,
        kd_pos=args.kd_pos,
        kp_ori=args.kp_ori,
        ki_ori=args.ki_ori,
        kd_ori=args.kd_ori,
        seed=args.seed
    )
    
    controller.run()


if __name__ == "__main__":
    main()
