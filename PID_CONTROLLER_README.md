# PID Home Position Controller for RoboCasa

This script implements a PID (Proportional-Integral-Derivative) controller to maintain the robot at its home position in the RoboCasa simulator.

## What is PID Control?

A PID controller is a feedback control mechanism that continuously calculates an error value as the difference between a desired setpoint (home position) and a measured process variable (current position). It then applies a correction based on three terms:

- **P (Proportional)**: Responds to the current error
- **I (Integral)**: Responds to accumulated past errors
- **D (Derivative)**: Responds to the rate of change of error

## Features

- ✅ **Position Control**: Maintains 3D end-effector position using PID
- ✅ **Orientation Control**: Maintains end-effector orientation using quaternion-based PID
- ✅ **Real-time Visualization**: Shows simulator and camera views
- ✅ **Error Logging**: Tracks position and orientation errors over time
- ✅ **Error Plotting**: Displays error graphs after execution
- ✅ **Tunable Gains**: Adjustable PID parameters via command-line arguments

## Usage

### Basic Usage (Default PID Gains)

```bash
./run_pid_control.sh
```

### Custom PID Gains

```bash
./run_pid_control.sh --kp-pos 2.0 --kd-pos 0.2 --kp-ori 1.0 --kd-ori 0.1
```

### All Available Options

```bash
python pid_home_position.py --help
```

**Arguments:**
- `--env`: Environment name (default: PnPCounterToCab)
- `--robot`: Robot model (default: PandaOmron)
- `--kp-pos`: Proportional gain for position (default: 1.0)
- `--ki-pos`: Integral gain for position (default: 0.0)
- `--kd-pos`: Derivative gain for position (default: 0.1)
- `--kp-ori`: Proportional gain for orientation (default: 0.5)
- `--ki-ori`: Integral gain for orientation (default: 0.0)
- `--kd-ori`: Derivative gain for orientation (default: 0.05)

## How It Works

1. **Initialization**: The environment is reset, and the robot's initial position is recorded as the "home position"
2. **Error Calculation**: At each time step, the controller computes:
   - Position error: `home_position - current_position`
   - Orientation error: Quaternion difference converted to axis-angle
3. **PID Computation**: The PID controller computes control actions based on the errors
4. **Action Execution**: The computed actions are sent to the robot
5. **Visualization**: Camera views update in real-time showing the robot maintaining its position
6. **Logging**: Position and orientation errors are logged for analysis

## PID Tuning Guide

### Position Control (kp_pos, ki_pos, kd_pos)

- **Increase kp_pos** (1.0 → 2.0): Faster response, but may cause oscillations
- **Increase kd_pos** (0.1 → 0.3): Reduces oscillations, dampens motion
- **Increase ki_pos** (0.0 → 0.01): Eliminates steady-state error, but can cause instability

### Orientation Control (kp_ori, ki_ori, kd_ori)

- **Increase kp_ori** (0.5 → 1.0): Faster orientation correction
- **Increase kd_ori** (0.05 → 0.1): Smoother orientation changes
- **ki_ori**: Usually kept at 0 for orientation control

### Recommended Starting Points

**Aggressive Control (Fast, May Oscillate):**
```bash
./run_pid_control.sh --kp-pos 2.0 --kd-pos 0.2 --kp-ori 1.0 --kd-ori 0.1
```

**Conservative Control (Slow, Stable):**
```bash
./run_pid_control.sh --kp-pos 0.5 --kd-pos 0.05 --kp-ori 0.3 --kd-ori 0.03
```

**Balanced Control (Default):**
```bash
./run_pid_control.sh --kp-pos 1.0 --kd-pos 0.1 --kp-ori 0.5 --kd-ori 0.05
```

## Testing the Controller

To test if the PID controller is working:

1. Run the script
2. Wait for the simulator and camera windows to appear
3. The robot should maintain its home position
4. You can manually perturb the robot in the simulator (if using MuJoCo viewer)
5. The PID controller should bring it back to the home position
6. Close the camera window to see error plots

## Output

The script provides:

1. **Real-time Console Output**: Position error, orientation error, and reward every 20 steps
2. **Camera Visualization**: Live camera feeds showing the robot
3. **Error Plots**: After closing, displays graphs of position and orientation errors over time

## Example Output

```
Creating PnPCounterToCab environment...
Resetting environment...

Home position set:
  Position: [0.123, -0.456, 0.789]
  Quaternion: [0.707, 0.0, 0.707, 0.0]

Task: pick the potato from the counter and place it in the cabinet
Available cameras: ['robot0_agentview_center', 'robot0_eye_in_hand', 'robot0_frontview']
Action space: (12,)

Starting simulator window...
Starting camera views window with PID control...
Step 0, Pos Error: 0.000123m, Ori Error: 0.000045rad, Reward: 0.0248
Step 20, Pos Error: 0.000089m, Ori Error: 0.000032rad, Reward: 0.0251
...
```

## Notes

- The controller uses the robot's end-effector position and orientation from the observation
- Gripper is kept closed (neutral position)
- Mobile base is kept stationary
- The home position is set to wherever the robot is after environment reset
- Anti-windup is implemented to prevent integral term from growing unbounded

## Troubleshooting

**Robot oscillates too much:**
- Decrease `kp_pos` and `kp_ori`
- Increase `kd_pos` and `kd_ori`

**Robot moves too slowly:**
- Increase `kp_pos` and `kp_ori`

**Robot doesn't return to exact position:**
- Add small `ki_pos` (e.g., 0.01)
- Increase `kp_pos`

**Controller is unstable:**
- Decrease all gains by 50%
- Start with only P control (set ki and kd to 0)
- Gradually add D term, then I term if needed
