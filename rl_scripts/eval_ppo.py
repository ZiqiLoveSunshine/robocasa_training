"""
PPO Evaluation Script for Robocasa PnPCounterToCab
"""

import argparse
import os
import sys
import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import PPO
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.controllers import load_composite_controller_config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import MyPnPCounterToCab
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PnPCounterToCab")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model zip")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--save_video", action="store_true", help="Save video of evaluation")
    parser.add_argument("--video_path", type=str, default="eval_videos", help="Directory to save videos")
    parser.add_argument("--reward_shaping", action="store_true", help="Enable dense rewards during evaluation")
    args = parser.parse_args()

    # Environment for evaluation (enable renderer if saving video)
    # Note: For video saving we need offscreen renderer
    has_offscreen = args.save_video
    
    robots = "PandaOmron" # Default robot
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots,
    )
    env = MyPnPCounterToCab(
        robots=robots,
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=has_offscreen,
        reward_shaping=args.reward_shaping, # Eval on sparse or dense? usually sparse for success rate
        control_freq=20,
        ignore_done=False,
        horizon=500,
        camera_names="robot0_eye_in_hand",
        camera_heights=512,
        camera_widths=512,
        render_camera="robot0_eye_in_hand",
    )
    
    env = GymWrapper(env, keys=None)
    
    model = PPO.load(args.model_path)
    
    if args.save_video:
        os.makedirs(args.video_path, exist_ok=True)

    success_count = 0
    
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        frames = []
        episode_reward = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if args.save_video:
                # Need to access underlying Robosuite env to render
                # GymWrapper -> env
                # But GymWrapper.step() returns obs, reward, done, etc.
                # To render we call env.sim.render(...) or similar? 
                # Actually robosuite envs have a .render() method but it might act differently.
                # Simplest way for offscreen:
                img = env.env.sim.render(
                    camera_name="robot0_eye_in_hand", 
                    width=512, 
                    height=512, 
                    depth=False
                )
                # Flip vertically because mujoco renders upside down
                img = np.flipud(img)
                frames.append(img)
        
        # Check success (if info has it, otherwise we might need custom check)
        # Note: Robosuite infos usually don't have 'is_success' standardly in GymWrapper unless added
        # But we can check internal logic if needed. 
        # For now, let's assume if return > threshold it might be good, 
        # OR we can manually check environment success
        is_success = env.env._check_success()
        if is_success:
            success_count += 1
            
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Success = {is_success}")
        
        if args.save_video and len(frames) > 0:
            vid_path = os.path.join(args.video_path, f"eval_ep_{ep}.mp4")
            imageio.mimsave(vid_path, frames, fps=20)
            print(f"Saved video to {vid_path}")

    print(f"Success Rate: {success_count}/{args.episodes} ({success_count/args.episodes*100:.2f}%)")
    env.close()

if __name__ == "__main__":
    main()
