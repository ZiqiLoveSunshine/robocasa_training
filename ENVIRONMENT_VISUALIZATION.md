# PnPCounterToCab Environment Visualization

## Overview
This document provides a comprehensive visualization and explanation of the **PnPCounterToCab** robotic manipulation task environment.

## Task Description
**PnPCounterToCab** is a single-stage atomic pick-and-place task where a robot must:
- Pick up a target object from a kitchen counter
- Place it inside an open cabinet

Example task: *"Pick the lime from the counter and place it in the cabinet"*

## Environment Components

### 1. Robot Configuration
- **Robot Type**: PandaOmron (Mobile manipulator)
  - Franka Panda robotic arm mounted on Omron mobile base
  - Provides both manipulation and mobility capabilities
- **Action Space**: 12-dimensional continuous control
  - Range: [-1.0, 1.0]
  - Control frequency: 20 Hz

### 2. Objects in Scene (3 total)
1. **Target Object** (e.g., lime, apple, can)
   - Placed on counter in front area
   - Must be graspable
   - Randomly sampled from object groups
   
2. **Distractor on Counter**
   - Placed in back area of counter
   - Tests robot's ability to distinguish target
   
3. **Distractor in Cabinet**
   - Placed in back of cabinet
   - Adds complexity to placement task

### 3. Kitchen Fixtures
- **Counter**: Where target object is initially placed
  - Front area: Target object placement zone
  - Back area: Distractor placement zone
  
- **Cabinet**: Destination for target object
  - Door state: 90-100% open (fully accessible)
  - Contains distractor object in back area

## Success Criteria
The task is successful when **BOTH** conditions are met:
1. ✅ Object is inside the cabinet
2. ✅ Gripper is far from the object (released)

## Reward Function
The environment provides dense reward shaping to guide learning:

| Component | Formula | Value Range | Description |
|-----------|---------|-------------|-------------|
| **Reach Object** | 1 - tanh(5.0 × dist_gripper_obj) | [0, 1] | Encourages gripper to approach target |
| **Grasp Bonus** | +5.0 if grasped | 0 or 5 | Contact between object and gripper |
| **Reach Cabinet** | 1 - tanh(5.0 × dist_obj_cab) | [0, 1] | Encourages moving object to cabinet |
| **Place Bonus** | +5.0 if placed | 0 or 5 | Object inside cabinet |
| **Success Bonus** | +2.0 if complete | 0 or 2 | Task fully completed |
| **Maximum Total** | - | ~14.0 | Sum of all components |

## Camera Observations

### Available Cameras
1. **robot0_agentview_center** (Primary)
   - Third-person view from above/front
   - Shows overall scene layout
   
2. **robot0_eye_in_hand**
   - Wrist-mounted camera
   - Close-up view of gripper and objects
   
3. **robot0_frontview**
   - Front-facing view
   - Alternative perspective

### Image Specifications
- **Training Resolution**: 128×128 pixels
- **Visualization Resolution**: 512×512 pixels
- **Format**: RGB images (H, W, 3)
- **Preprocessing**: Transposed to (3, H, W) for PyTorch CNN

## Training Configuration

### Algorithm
- **Method**: Proximal Policy Optimization (PPO)
- **Observation Type**: Visual (image-based)
- **Action Type**: Continuous control

### CNN Architecture
```
Input: (3, 128, 128) RGB image
↓
Conv2D(3→32, kernel=8, stride=4) + ReLU
↓
Conv2D(32→64, kernel=4, stride=2) + ReLU
↓
Conv2D(64→64, kernel=3, stride=1) + ReLU
↓
Flatten → FC(256) + ReLU
↓
┌─────────────────┬─────────────────┐
│   Policy Head   │   Value Head    │
│  FC(128)+ReLU   │  FC(128)+ReLU   │
│  FC(12)+Tanh    │     FC(1)       │
│  (actions)      │   (value est.)  │
└─────────────────┴─────────────────┘
```

### Key Hyperparameters
- **Learning Rate**: 5e-4 (with KL-adaptive scheduling)
- **Discount Factor (γ)**: 0.9
- **GAE Lambda (λ)**: 0.95
- **Clip Range**: 0.2
- **Mini-batches**: 32
- **Learning Epochs**: 10 per update
- **Rollout Steps**: Adaptive based on number of environments

## Visualization Files Generated

1. **pnp_counter_to_cab_visualization.png**
   - Grid of 6 camera views showing the environment
   - Includes views from different cameras and timesteps
   - Shows robot, objects, counter, and cabinet

2. **pnp_environment_visualization.html**
   - Interactive HTML visualization
   - Animated diagram of environment layout
   - Detailed task information and reward breakdown
   - Can be opened in any web browser

3. **visualize_env.py**
   - Python script to generate visualizations
   - Can be run to create new visualizations with different seeds
   - Usage: `python visualize_env.py --output <filename>.png`

## Task Complexity Analysis

### Challenges
1. **Visual Perception**: Must identify target object among distractors
2. **Precise Manipulation**: Requires accurate grasping and placement
3. **Spatial Reasoning**: Navigate from counter to cabinet
4. **Partial Observability**: Limited field of view from cameras
5. **High-Dimensional Control**: 12D continuous action space

### Learning Progression
Typical learning stages:
1. **Exploration**: Random movements, rarely touching objects
2. **Reaching**: Learning to move gripper toward objects
3. **Grasping**: Developing reliable grasp strategies
4. **Transport**: Moving grasped objects toward cabinet
5. **Placement**: Precisely placing objects inside cabinet
6. **Release**: Learning to release object and retract

## Environment Variations
The environment randomizes:
- Target object type (from specified object groups)
- Object positions (within defined regions)
- Distractor objects
- Kitchen layout (multiple possible layouts)
- Initial robot position (near cabinet)

This ensures the policy generalizes rather than memorizing specific configurations.

## Code Implementation Details

### Environment Creation
```python
env = robosuite.make(
    env_name="PnPCounterToCab",
    robots="PandaOmron",
    has_renderer=False,           # Headless training
    has_offscreen_renderer=True,  # Enable camera rendering
    use_camera_obs=True,          # Visual observations
    camera_names=["robot0_agentview_center"],
    camera_heights=128,
    camera_widths=128,
    control_freq=20,
    reward_shaping=True,
)
```

### Observation Wrapper
The `RobocasaImageWrapper` class:
- Extracts image observations from Robosuite's dict observations
- Transposes images from (H, W, C) to (C, H, W) for PyTorch
- Converts action space to Gymnasium format
- Handles reset and step methods

## Performance Metrics

During training, monitor:
- **Episode Return**: Total reward accumulated
- **Success Rate**: Percentage of successful task completions
- **Episode Length**: Number of steps per episode
- **Reward Components**: Individual reward terms to diagnose learning
- **Action Statistics**: Mean, std, min, max of actions

## References

- **RoboCasa**: Kitchen manipulation benchmark
- **Robosuite**: Robot simulation framework
- **SKRL**: Reinforcement learning library
- **MuJoCo**: Physics engine (with EGL rendering)

---

Generated on: 2026-02-02
Environment: PnPCounterToCab
Robot: PandaOmron
Framework: RoboCasa + Robosuite + SKRL
