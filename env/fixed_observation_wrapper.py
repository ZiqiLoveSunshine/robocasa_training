"""
Custom wrapper to fix observation space mismatch in RoboCasa environments.
"""
import gymnasium as gym
import numpy as np


class FixedObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to fix observation space mismatch in RoboCasa's GymWrapper.
    
    The RoboCasa GymWrapper has a bug where the declared observation space
    doesn't match the actual observations returned. This wrapper fixes that
    by determining the actual observation space from a reset and updating
    the declared space accordingly.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Get actual observation from a reset to determine true observation space
        actual_obs, _ = env.reset()
        actual_shape = actual_obs.shape
        
        # Update observation space to match actual observations
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=actual_shape,
            dtype=np.float32
        )
        
        print(f"[FixedObservationWrapper] Corrected observation space to {actual_shape}")
    
    def observation(self, observation):
        """Pass through the observation unchanged."""
        return observation
