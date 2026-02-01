
import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

# reuse imports and setup from train.py
# This ensures sys.path is set correctly if train.py does it at module level
import train 
from train import Policy, Value, create_sim_env, RobocasaImageWrapper, set_seed, CNN

import gymnasium
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env

def run_eval(args):
    # Set seed
    set_seed(args.seed)
    
    print(f"Creating environment: {args.env_name}")
    # Initialize environment
    # create_sim_env in train.py sets camera_obs=True, etc.
    env = create_sim_env(args.env_name, seed=args.seed)
    
    # Wrap for SKRL compatibility (converts to Tensor, Handles batch dimension)
    env = wrap_env(env)
    
    # Force device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Instantiate models
    # We must instantiate the exact same model structure
    models = {}
    models["policy"] = Policy(env.observation_space, env.state_space, env.action_space, device)
    models["value"] = Value(env.observation_space, env.state_space, env.action_space, device) 

    # 3. Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        # SKRL PPO.load() handles loading the checkpoint file
    else:
        print("WARNING: No checkpoint provided! Running with random policy.")

    # 4. Instantiate Agent
    cfg = PPO_CFG()
    cfg.experiment.write_interval = 0 
    cfg.experiment.checkpoint_interval = 0
    # Disable preprocessors if train.py disabled them (it did, for images)
    cfg.observation_preprocessor = None
    cfg.state_preprocessor = None
    
    agent = PPO(
        models=models,
        memory=None, 
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
    )
    
    if args.checkpoint:
        agent.load(args.checkpoint)
        
    # 5. Evaluation Loop
    print(f"Starting evaluation for {args.max_steps} steps...")
    
    obs, _ = env.reset()
    
    frames = []
    total_reward = 0
    
    for step in range(args.max_steps):
        # Action
        with torch.no_grad():
            # agent.act(states, ...) matches skrl API
            # signature: act(self, observations, states, timestep, timesteps)
            # We pass states=None as we use pixel obs only and policy handles it
            actions = agent.act(obs, None, timestep=step, timesteps=args.max_steps)[0]
            
        # Step
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Track reward
        # reward is a tensor
        total_reward += reward.item()
        
        if args.record_video:
            # Debug shape
            if step == 0:
                print(f"DEBUG: obs type: {type(obs)}")
                if isinstance(obs, torch.Tensor):
                    print(f"DEBUG: obs shape: {obs.shape}")
                elif isinstance(obs, np.ndarray):
                    print(f"DEBUG: obs shape: {obs.shape}")
            
            img_tensor = obs
            
            # Handle flattened observation (SKRL wrapper might flatten)
            if isinstance(img_tensor, torch.Tensor):
                if img_tensor.dim() == 2: # (B, F) e.g. (1, 21168)
                    # Reshape to (B, C, H, W)
                    img_tensor = img_tensor.view(-1, 3, 84, 84)
                
                if img_tensor.dim() == 4: # (B, C, H, W)
                     img_tensor = img_tensor[0]
                img_np = img_tensor.cpu().numpy().astype(np.uint8)
            else:
                img_np = img_tensor # Assume numpy
            
            # Now (C, H, W) check
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0)) # C,H,W -> H,W,C
            elif img_np.ndim == 3 and img_np.shape[2] == 3:
                pass # Already H,W,C
            else:
                print(f"WARNING: Unexpected image shape: {img_np.shape}")
                continue # Skip frame

            frames.append(img_np)
            
        if terminated.any() or truncated.any():
            print(f"Episode finished at step {step+1} with total reward {total_reward}")
            # Reset is handled automatically by SKRL wrapper?
            # Usually wrap_env wraps with AutoResetWrapper logic or similar?
            # SKRL's wrap_env usually keeps going.
            # But let's break if we just want one episode viz
            break
            
    print("Evaluation finished.")
    
    if args.record_video and len(frames) > 0:
        import imageio
        video_filename = f"eval_{args.env_name}.mp4"
        print(f"Saving video to {video_filename} ({len(frames)} frames)...")
        imageio.mimsave(video_filename, frames, fps=20)
        print("Video saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="PnPCounterToCab")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (agent.pt)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--record_video", action="store_true", help="Record video of evaluation")
    
    # default record_video to true if not specified? 
    # argparse 'store_true' implies False by default.
    # Let's make it easy for user
    
    args = parser.parse_args()
    
    # If user didn't specify record_video, let's assume they might want it if headless?
    # No, let's stick to flags.
    
    run_eval(args)
