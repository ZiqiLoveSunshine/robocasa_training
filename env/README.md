# Custom PnPCounterToCab Environment

This folder contains a custom implementation of the PnPCounterToCab environment that allows you to modify the reward function without touching the original `robocasa` or `skrl` packages.

## Structure

```
env/
├── __init__.py                      # Package initialization
└── custom_pnp_counter_to_cab.py     # Custom environment with modifiable reward
```

## Features

- **Inheritance-based**: Inherits from the original `PnPCounterToCab` class
- **Non-invasive**: No modifications to robocasa or skrl packages
- **Flexible**: Easy to swap between different reward functions
- **Examples included**: Multiple reward function variants provided

## Usage

### Basic Usage

```python
from env import MyPnPCounterToCab
from robosuite.controllers import load_composite_controller_config

# Setup controller
robots = "PandaOmron"
controller_config = load_composite_controller_config(
    controller=None,
    robot=robots,
)

# Create custom environment
env = MyPnPCounterToCab(
    robots=robots,
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["robot0_agentview_center"],
    camera_heights=128,
    camera_widths=128,
    control_freq=20,
    reward_shaping=True,
)

# Use like any other environment
obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

### Using with Training Script

Modify your `train.py` to use the custom environment:

```python
# Add this import at the top
from env import MyPnPCounterToCab

# In create_sim_env function, replace:
# env = robosuite.make(**env_kwargs)
# with:
env = MyPnPCounterToCab(**env_kwargs)
```

Or use robosuite.make by passing the class directly:

```python
import robosuite
from env import MyPnPCounterToCab

env = robosuite.make(
    env_class=MyPnPCounterToCab,  # Use custom class
    # ... other arguments
)
```

## Available Reward Functions

The `MyPnPCounterToCab` class includes several reward function implementations:

### 1. `reward()` - Default (Original)
The standard reward function from RoboCasa:
- Reach object: distance-based (max ~1.0)
- Grasp bonus: +5.0
- Reach cabinet: distance-based (max ~1.0)
- Place bonus: +5.0
- Success bonus: +2.0
- **Total max: ~14.0**

### 2. `reward_sparse()` - Sparse Reward
Only rewards on task completion:
- Success: +10.0
- Otherwise: 0.0
- **Harder to learn but no shaping bias**

### 3. `reward_dense_alternative()` - Alternative Dense
Different shaping with exponential decay:
- Reach object: exp(-2.0 × distance)
- Grasp bonus: +3.0
- Move to cabinet: exp(-2.0 × distance) (only if grasped)
- Place bonus: +10.0
- Success bonus: +5.0

### 4. `reward_curriculum()` - Curriculum-based
Template for curriculum learning (requires episode tracking)

### 5. `reward_with_penalties()` - With Penalties
Original reward plus penalties for:
- Large actions (smoothness)
- Touching distractor objects

## Modifying the Reward Function

### Option 1: Edit the Default Reward

Edit `env/custom_pnp_counter_to_cab.py` and modify the `reward()` method:

```python
def reward(self, action=None):
    reward = 0.0
    
    # Your custom reward logic here
    obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]]
    gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
    
    # Example: Simple distance-based reward
    dist = np.linalg.norm(gripper_pos - obj_pos)
    reward = -dist  # Negative distance
    
    return reward
```

### Option 2: Use an Alternative Reward

Rename one of the alternative reward functions to `reward`:

```python
# In custom_pnp_counter_to_cab.py

# Rename current reward
def reward_original(self, action=None):
    # ... original implementation

# Rename alternative to reward
def reward(self, action=None):  # Was: reward_sparse
    if self._check_success():
        return 10.0
    return 0.0
```

### Option 3: Create a New Reward Function

Add a new method and rename it to `reward`:

```python
def reward_my_custom(self, action=None):
    """My custom reward function."""
    reward = 0.0
    
    # Access environment state
    obj_pos = self.sim.data.body_xpos[self.obj_body_id["obj"]]
    gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
    cab_pos = self.cab.pos
    
    # Your custom logic
    # ...
    
    return reward

# Then rename it to 'reward' to use it
```

## Available Environment Attributes

When writing custom rewards, you have access to:

### Object Information
- `self.objects` - Dictionary of all objects
- `self.obj_body_id` - Body IDs for objects
- `self.objects["obj"]` - Target object
- `self.objects["distr_counter"]` - Counter distractor
- `self.objects["distr_cab"]` - Cabinet distractor

### Robot Information
- `self.robots[0]` - Robot instance
- `self.robots[0].eef_site_id["right"]` - End-effector site ID
- `self.robots[0].gripper` - Gripper instance

### Fixture Information
- `self.cab` - Cabinet fixture
- `self.counter` - Counter fixture
- `self.cab.pos` - Cabinet position

### Simulation Data
- `self.sim.data.body_xpos[id]` - Body position
- `self.sim.data.site_xpos[id]` - Site position
- `self.sim.data.qpos` - Joint positions
- `self.sim.data.qvel` - Joint velocities

### Utility Functions
- `self.check_contact(obj1, obj2)` - Check contact between objects
- `self._check_success()` - Check if task is successful
- `OU.obj_inside_of(self, "obj", fixture)` - Check if object inside fixture
- `OU.gripper_obj_far(self)` - Check if gripper far from object

## Testing

Test your custom environment:

```bash
# Basic test
python test_custom_env.py

# Compare with original
python test_custom_env.py --compare
```

## Integration with Training

To use the custom environment in your training pipeline:

1. **Modify `train.py`**:
   ```python
   from env import MyPnPCounterToCab
   
   # In create_sim_env function:
   env = MyPnPCounterToCab(**env_kwargs)
   ```

2. **Run training**:
   ```bash
   python train.py --env_name PnPCounterToCab --max_timesteps 500000
   ```

The environment name is just for logging; the actual environment used is `MyPnPCounterToCab`.

## Tips for Reward Design

1. **Start with the original**: Understand the baseline before modifying
2. **Test incrementally**: Make small changes and test
3. **Monitor learning**: Track reward components separately
4. **Balance sparse/dense**: Dense rewards help learning, but can introduce bias
5. **Avoid reward hacking**: Make sure rewards align with true objective
6. **Scale appropriately**: Keep reward magnitudes reasonable (e.g., -10 to +10)

## Example: Switching to Sparse Reward

```python
# In env/custom_pnp_counter_to_cab.py

# Comment out or rename the current reward
def reward_dense(self, action=None):
    # ... original dense reward

# Rename sparse to reward
def reward(self, action=None):
    """Sparse reward: only on success."""
    if self._check_success():
        return 10.0
    return 0.0
```

Then train as normal:
```bash
python train.py --env_name PnPCounterToCab --max_timesteps 1000000
```

## Troubleshooting

### Import Error
If you get import errors, make sure:
- The `env/` folder is in your project root
- `__init__.py` exists in the `env/` folder
- You're running from the project root directory

### Reward Not Changing
Make sure you:
- Modified the `reward()` method (not an alternative)
- Saved the file
- Restarted your training script

### Environment Not Found
If using `robosuite.make()`, pass the class directly:
```python
env = robosuite.make(env_class=MyPnPCounterToCab, ...)
```

## Advanced: Multiple Custom Environments

You can create multiple custom environments:

```python
# env/custom_pnp_sparse.py
class CustomPnPSparse(PnPCounterToCab):
    def reward(self, action=None):
        return 10.0 if self._check_success() else 0.0

# env/custom_pnp_dense.py
class CustomPnPDense(PnPCounterToCab):
    def reward(self, action=None):
        # Dense reward implementation
        pass

# env/__init__.py
from env.custom_pnp_sparse import CustomPnPSparse
from env.custom_pnp_dense import CustomPnPDense
```

Then switch between them in your training script.

## References

- [RoboCasa Documentation](https://robocasa.ai/)
- [Robosuite Documentation](https://robosuite.ai/)
- Original reward function: `robocasa/environments/kitchen/single_stage/kitchen_pnp.py`
