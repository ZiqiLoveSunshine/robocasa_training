# Custom Reward Function - Quick Start Guide

## 📁 Files Created

```
env/
├── __init__.py                      # Package initialization
├── custom_pnp_counter_to_cab.py     # Custom environment with reward functions
└── README.md                        # Detailed documentation

train_custom.py                      # Training script using custom environment
test_custom_env.py                   # Test script for custom environment
```

## 🚀 Quick Start

### 1. Test the Custom Environment

```bash
# Basic test
python test_custom_env.py

# Compare with original environment
python test_custom_env.py --compare
```

### 2. Train with Custom Environment

```bash
# Single environment
python train_custom.py --env_name PnPCounterToCab --max_timesteps 500000

# Multiple parallel environments
python train_custom.py --env_name PnPCounterToCab --max_timesteps 500000 --num_envs 4

# With WandB logging
python train_custom.py --env_name PnPCounterToCab --max_timesteps 500000 --num_envs 4 --wandb
```

## ✏️ Modify the Reward Function

### Option 1: Edit the Default Reward (Recommended)

Edit `env/custom_pnp_counter_to_cab.py`, find the `reward()` method (around line 50):

```python
def reward(self, action=None):
    """
    Custom reward function - EDIT THIS!
    """
    reward = 0.0
    
    # Get positions
    obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]]
    gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
    
    # YOUR CUSTOM REWARD LOGIC HERE
    # Example: Simple distance-based reward
    dist = np.linalg.norm(gripper_pos - obj_pos)
    reward = -dist
    
    return reward
```

### Option 2: Use a Pre-made Alternative

The file includes several alternative reward functions:
- `reward_sparse()` - Only reward on success
- `reward_dense_alternative()` - Different dense shaping
- `reward_with_penalties()` - Includes action penalties

To use one, rename it to `reward()`:

```python
# In env/custom_pnp_counter_to_cab.py

# Rename current reward
def reward_original(self, action=None):
    # ... original code

# Rename alternative to reward
def reward(self, action=None):  # Was: reward_sparse
    if self._check_success():
        return 10.0
    return 0.0
```

## 📊 Available Reward Functions

| Function | Description | Max Reward |
|----------|-------------|------------|
| `reward()` (default) | Original dense reward | ~14.0 |
| `reward_sparse()` | Only on success | 10.0 |
| `reward_dense_alternative()` | Exponential decay | ~18.0 |
| `reward_with_penalties()` | Original + penalties | ~14.0 |

## 🔍 Accessing Environment State

In your custom reward function, you can access:

```python
# Object positions
obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]]
gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
cab_pos = self.cab.pos

# Check contacts
is_grasped = self.check_contact(self.objects["obj"], self.robots[0].gripper)

# Check success
is_success = self._check_success()

# Check if object in cabinet
is_placed = OU.obj_inside_of(self, "obj", self.cab)
```

## 🎯 Example Modifications

### Example 1: Sparse Reward Only

```python
def reward(self, action=None):
    return 10.0 if self._check_success() else 0.0
```

### Example 2: Distance-Based Only

```python
def reward(self, action=None):
    obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]]
    gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
    
    dist = np.linalg.norm(gripper_pos - obj_pos)
    return -dist  # Negative distance
```

### Example 3: Staged Reward

```python
def reward(self, action=None):
    reward = 0.0
    obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]]
    gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
    
    # Stage 1: Reach object
    dist_to_obj = np.linalg.norm(gripper_pos - obj_pos)
    if dist_to_obj < 0.1:
        reward += 5.0
        
        # Stage 2: Grasp
        if self.check_contact(self.objects["obj"], self.robots[0].gripper):
            reward += 10.0
            
            # Stage 3: Move to cabinet
            dist_to_cab = np.linalg.norm(obj_pos - self.cab.pos)
            if dist_to_cab < 0.2:
                reward += 15.0
                
                # Stage 4: Place
                if OU.obj_inside_of(self, "obj", self.cab):
                    reward += 20.0
    
    return reward
```

## 🔄 Workflow

1. **Edit** `env/custom_pnp_counter_to_cab.py`
2. **Test** with `python test_custom_env.py`
3. **Train** with `python train_custom.py --max_timesteps 500000`
4. **Monitor** training progress
5. **Iterate** - modify reward and repeat

## ⚠️ Important Notes

- **Don't modify** files in `robocasa/` or `skrl/` folders
- **Always test** your reward function before training
- **Save backups** before making major changes
- **Monitor learning** - if reward doesn't increase, adjust shaping
- **Start simple** - test with sparse reward first, then add shaping

## 🐛 Troubleshooting

### Reward is always 0
- Check if you're accessing the right variables
- Print debug info: `print(f"Reward: {reward}, Success: {self._check_success()}")`

### Training not improving
- Reward might be too sparse - add more shaping
- Check if reward scale is appropriate (not too large/small)
- Verify success criteria is achievable

### Import errors
- Make sure you're running from project root
- Check that `env/__init__.py` exists
- Verify paths in imports

## 📚 Further Reading

- Full documentation: `env/README.md`
- Original reward: `robocasa/robocasa/environments/kitchen/single_stage/kitchen_pnp.py` (line 142)
- Environment details: `ENVIRONMENT_VISUALIZATION.md`

## 💡 Tips

1. **Start with original reward** - understand baseline first
2. **Test incrementally** - small changes at a time
3. **Compare alternatives** - use `test_custom_env.py --compare`
4. **Log reward components** - track each part separately
5. **Visualize** - use `visualize_env.py` to see what's happening

---

**Ready to start?** Edit `env/custom_pnp_counter_to_cab.py` and run `python train_custom.py`!
