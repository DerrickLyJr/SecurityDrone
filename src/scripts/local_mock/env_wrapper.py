# /home/dlyerly/SecurityDrone/src/scripts/local_mock/env_wrapper.py
import torch

class SkrlVecEnvWrapper:
    def __init__(self, env):
        self.env = env
        
        # --- 1. CORE SPACES & DIMENSIONS ---
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = env.num_envs
        
        # --- 2. HARDWARE & TRACKING GATEKEEPERS ---
        self.device = env.device if hasattr(env, 'device') else torch.device("cpu")
        self.state_space = env.observation_space # Identical to obs space for single-agent PPO

    # --- 3. MANDATORY SKRL METHODS ---
    def reset(self):
        """Called at step 0 to initialize tracking tensors."""
        obs, infos = self.env.reset()
        return obs, infos

    def step(self, actions):
        """Called every optimization rollout frame."""
        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        return next_obs, rewards, terminated, truncated, infos

    def state(self):
        """
        🔥 FIXED: Called by SKRL trainers to fetch the global state tensor.
        For single-agent setups like tracking a target, this is just your 
        current observation vector.
        """
        return self.env.compute_observations()

    def close(self):
        """Called when training finishes or crashes to release simulation memory."""
        if hasattr(self.env, "close"):
            self.env.close()

    # --- 4. THE ULTIMATE BACKSTOP (No More AttributeErrors) ---
    def __getattr__(self, name):
        """
        Catches ANY missing property or method call from SKRL and forwards 
        it directly to your TelloAdaptiveEnv.
        """
        return getattr(self.env, name)
    
    def render(self):
        """Placeholder to prevent Gymnasium/SKRL from crashing during training."""
        pass