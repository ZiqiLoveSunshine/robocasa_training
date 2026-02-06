# Training and Evaluation Summary

## Successfully Created Files

### 1. Custom Environment: `MyTurnOnMicrowave`
**Location:** `/home/ziqi/robocasa_training/env/custom_turn_on_microwave.py`

**Features:**
- Custom reward function for the TurnOnMicrowave task
- Dense reward shaping with 5 components:
  1. **Reach Reward** (weight: 1.0) - Exponential decay based on distance to button
  2. **Press Reward** (weight: 10.0) - Large bonus for turning on microwave
  3. **Retreat Reward** (weight: 1.0) - Encourages moving away after pressing
  4. **Success Reward** (weight: 5.0) - Bonus for complete task success
  5. **Action Penalty** (weight: 0.01) - Encourages smooth movements

**Alternative Reward Functions:**
- `reward_sparse` - Pure sparse reward
- `reward_simple_dense` - Simpler dense reward
- `reward_staged` - Stage-based rewards
- `reward_with_velocity_penalty` - Additional velocity penalties

### 2. Fixed Observation Wrapper
**Location:** `/home/ziqi/robocasa_training/env/fixed_observation_wrapper.py`

**Purpose:** Fixes a bug in RoboCasa's GymWrapper where the declared observation space doesn't match actual observations.

### 3. Updated Training Script
**Location:** `/home/ziqi/robocasa_training/rl_scripts/train_ppo.py`

**Changes:**
- Added support for `TurnOnMicrowave` task
- Added `FixedObservationWrapper` to handle observation space mismatch
- Fixed vectorized environment creation
- Uses `keys=None` for GymWrapper due to RoboCasa bug

### 4. Updated Evaluation Script
**Location:** `/home/ziqi/robocasa_training/rl_scripts/eval_ppo.py`

**Changes:**
- Added support for both `PnPCounterToCab` and `TurnOnMicrowave` tasks
- Uses same environment configuration as training script
- Includes `FixedObservationWrapper` for consistency
- Task selection via `--task` argument

## Usage

### Training

#### PnPCounterToCab (with 8 parallel environments):
```bash
conda activate robocasa
python rl_scripts/train_ppo.py --task PnPCounterToCab --max_timesteps 3000000 --n_envs 8
```

#### TurnOnMicrowave (with 1 environment due to observation space issues):
```bash
conda activate robocasa
python rl_scripts/train_ppo.py --task TurnOnMicrowave --max_timesteps 3000000 --n_envs 1
```

**Note:** TurnOnMicrowave currently only supports `n_envs=1` due to non-deterministic observation space dimensions across different environment instances (different objects spawn with different observation dimensions).

### Evaluation

#### Evaluate PnPCounterToCab:
```bash
conda activate robocasa
python rl_scripts/eval_ppo.py \\
    --task PnPCounterToCab \\
    --model_path models/PnPCounterToCab_ppo_YYYYMMDD_HHMMSS/final_model \\
    --episodes 10 \\
    --save_video
```

#### Evaluate TurnOnMicrowave:
```bash
conda activate robocasa
python rl_scripts/eval_ppo.py \\
    --task TurnOnMicrowave \\
    --model_path models/TurnOnMicrowave_ppo_YYYYMMDD_HHMMSS/final_model \\
    --episodes 10 \\
    --save_video
```

### Evaluation Options:
- `--task`: Task name (`PnPCounterToCab` or `TurnOnMicrowave`)
- `--model_path`: Path to trained model (required)
- `--episodes`: Number of episodes to evaluate (default: 5)
- `--save_video`: Save video of evaluation episodes
- `--video_path`: Directory to save videos (default: `eval_videos`)
- `--reward_shaping`: Enable dense rewards during evaluation (default: False for sparse)
- `--seed`: Random seed (default: 100)

## Key Issues Resolved

### 1. Observation Space Mismatch
**Problem:** RoboCasa's GymWrapper has a bug where:
- With `keys=None`: Declares observation space as (96,) but returns (82,)
- With explicit keys: Declares observation space as (82,) but returns (96,)

**Solution:** Created `FixedObservationWrapper` that:
1. Performs an initial reset to determine actual observation shape
2. Updates the declared observation space to match actual observations
3. Ensures compatibility with Stable-Baselines3's vectorized environments

### 2. Variable Observation Dimensions
**Problem:** TurnOnMicrowave environment spawns random objects, leading to different observation dimensions across environment instances.

**Solution:** Currently limited to `n_envs=1`. For multi-environment training, would need to:
- Fix object spawning to be deterministic
- Or use only robot proprioception (no object states)
- Or implement observation padding/masking

### 3. Button Position Retrieval
**Problem:** Initial implementation used non-existent `get_button_pos()` method.

**Solution:** Use MuJoCo's geom API:
```python
button_id = self.sim.model.geom_name2id(
    "{}start_button".format(self.microwave.naming_prefix)
)
button_pos = self.sim.data.geom_xpos[button_id]
```

## Monitoring Training

Training metrics are logged to Weights & Biases:
- Project: `Graph-Mobile-Manipulator`
- Run name format: `{TaskName}_ppo_{YYYYMMDD_HHMMSS}`

View runs at: https://wandb.ai/Graph-Mobile-Manipulator/Graph-Mobile-Manipulator

## File Structure

```
robocasa_training/
├── env/
│   ├── __init__.py                      # Exports custom environments
│   ├── custom_pnp_counter_to_cab.py    # Custom PnPCounterToCab
│   ├── custom_turn_on_microwave.py     # Custom TurnOnMicrowave ✨ NEW
│   └── fixed_observation_wrapper.py    # Observation space fix ✨ NEW
├── rl_scripts/
│   ├── train_ppo.py                     # Training script (updated) ✨
│   └── eval_ppo.py                      # Evaluation script (updated) ✨
├── test_turn_on_microwave.py           # Test script ✨ NEW
└── script.sh                            # Training commands
```

## Next Steps

1. **Test Training:** Run the TurnOnMicrowave training to verify it works
2. **Monitor Progress:** Check WandB for training metrics
3. **Evaluate:** Once training completes, run evaluation script
4. **Tune Rewards:** Adjust reward weights in `custom_turn_on_microwave.py` if needed
5. **Try Alternative Rewards:** Experiment with different reward functions

## Troubleshooting

### If training fails with observation space errors:
- Ensure `FixedObservationWrapper` is applied after `GymWrapper`
- Check that `keys=None` is used in `GymWrapper`
- Verify `n_envs=1` for TurnOnMicrowave

### If reward seems wrong:
- Check reward function in `custom_turn_on_microwave.py`
- Verify button position retrieval is working
- Test with `test_turn_on_microwave.py`

### If evaluation fails:
- Ensure model path is correct
- Verify task name matches training task
- Check that same wrappers are used as in training
