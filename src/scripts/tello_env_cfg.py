import torch
import gymnasium as gym
from collections import deque

class TelloAdaptiveEnv(gym.Env):
    def __init__(self, num_envs=2048, device="cuda"):
        super().__init__()
        self.num_envs = num_envs
        self.device = device
        
        # Default control mode (can be toggled to "waypoint" later)
        self.control_mode = "velocity" 
        
        # Dimensions (7 features for hybrid action layer)
        self.observation_space_dim = 6
        self.action_space_dim = 7  # Expanded to handle hybrid space natively

        self.observation_space = gym.spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(self.observation_space_dim,), 
            dtype=float
        )
        
        # 7 actions: [roll, pitch, throttle, yaw, X_tgt, Y_tgt, Z_tgt]
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.action_space_dim,), 
            dtype=float
        )
        
        # --- PERFORMANCE-BASED TRACKING ---
        self.difficulty_factor = 0.0  
        self.success_window = deque(maxlen=100) 
        
        # --- LATENCY BUFFER ---
        self.max_latency_steps = 5    
        self.obs_history = []         
        
        # Ground truth buffers allocated directly on GPU
        self.drone_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drone_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.episode_steps = torch.zeros(self.num_envs, device=self.device)
        
        # Mock view placeholders (Isaac Lab injects the real views during instantiation)
        self.drone_view = None 
        self.target_view = None

    def _get_success_rate(self):
        if not self.success_window:
            return 0.0
        return sum(self.success_window) / len(self.success_window)

    def _update_curriculum(self):
        success_rate = self._get_success_rate()
        if success_rate > 0.85 and self.difficulty_factor < 1.0:
            self.difficulty_factor = min(1.0, self.difficulty_factor + 0.05)
            print(f"[CURRICULUM UP] Success Rate: {success_rate:.2f}. Difficulty bumped to {self.difficulty_factor:.2f}")
        elif success_rate < 0.50 and self.difficulty_factor > 0.0:
            self.difficulty_factor = max(0.0, self.difficulty_factor - 0.05)
            print(f"[CURRICULUM DOWN] Success Rate: {success_rate:.2f}. Difficulty reduced to {self.difficulty_factor:.2f}")

    def compute_observations(self):
        rel_pos = self.target_pos - self.drone_pos
        noise_amplitude = 0.15 * self.difficulty_factor
        
        pos_noise = torch.randn_like(rel_pos) * noise_amplitude
        vel_noise = torch.randn_like(self.drone_vel) * noise_amplitude
        
        noisy_rel_pos = rel_pos + pos_noise
        noisy_vel = self.drone_vel + vel_noise
        
        current_perfect_obs = torch.cat([noisy_rel_pos, noisy_vel], dim=-1)
        self.obs_history.append(current_perfect_obs.clone())
        
        current_delay = int(self.difficulty_factor * self.max_latency_steps)
        if len(self.obs_history) > current_delay:
            delayed_obs = self.obs_history.pop(0)
        else:
            delayed_obs = self.obs_history[0]
            
        return delayed_obs
    
    def reset(self, seed=None, options=None):
        """Resets the entire vector of environments globally at the start of an experiment."""
        # Standard Gymnasium seed initialization
        if seed is not None:
            super().reset(seed=seed)
        
        # 1. Reset all state tracking steps
        self.episode_steps.fill_(0)
        
        # 2. Reset Drones to their starting launch pad positions (0.0, 0.0, 1.0)
        self.drone_pos.fill_(0.0)
        self.drone_pos[:, 2] = 1.0  # Float hover altitude at 1 meter
        self.drone_vel.fill_(0.0)
        
        # 3. Reset Targets nearby using initial difficulty metrics
        max_spawn_radius = 2.0 + (self.difficulty_factor * 8.0)
        random_offsets = (torch.rand((self.num_envs, 2), device=self.device) * 2.0 - 1.0) * max_spawn_radius
        
        self.target_pos[:, 0] = self.drone_pos[:, 0] + random_offsets[:, 0]
        self.target_pos[:, 1] = self.drone_pos[:, 1] + random_offsets[:, 1]
        self.target_pos[:, 2] = 1.0  # Human standing height coordinate
        self.target_vel.fill_(0.0)
        
        # 4. Clear memory arrays inside your Wi-Fi latency buffer history
        self.obs_history.clear()
        
        # 5. Push initial configurations down to the PhysX core if views are initialized
        if self.drone_view is not None and self.target_view is not None:
            # Generate whole-buffer index masks matching all active environments [0, 1, ..., num_envs-1]
            all_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
            self.drone_view.set_world_poses(self.drone_pos, indices=all_indices)
            self.target_view.set_world_poses(self.target_pos, indices=all_indices)
            
        # 6. Compute initial observations state frame
        initial_obs = self.compute_observations()
        
        # 7. Return standard Gymnasium tuple structure
        return initial_obs, {"difficulty": self.difficulty_factor, "success_rate": self._get_success_rate()}

    def step(self, actions):
        self.episode_steps += 1
        
        # --- MOVING TARGET LOGIC ---
        if self.difficulty_factor > 0.2:
            target_evasion_speed = 2.0 * self.difficulty_factor 
            random_drift = (torch.rand_like(self.target_vel) * 2.0 - 1.0) * target_evasion_speed
            self.target_pos += random_drift * 0.016 
            
        actions = torch.clamp(actions, min=-1.0, max=1.0)
        physx_velocity_cmds = torch.zeros((self.num_envs, 6), device=self.device)
        
        if self.control_mode == "velocity":
            physx_velocity_cmds[:, 0] = actions[:, 1] * 2.0   # Pitch -> X
            physx_velocity_cmds[:, 1] = actions[:, 0] * 2.0   # Roll -> Y
            physx_velocity_cmds[:, 2] = actions[:, 2] * 1.5   # Throttle -> Z
            physx_velocity_cmds[:, 5] = actions[:, 3] * 3.14  # Yaw -> Spin Z
        elif self.control_mode == "waypoint":
            local_target_waypoints = actions[:, 4:7] * 5.0 
            time_to_reach = 2.0  
            physx_velocity_cmds[:, 0:3] = local_target_waypoints / time_to_reach
            physx_velocity_cmds[:, 5] = 0.0 
            
        if self.drone_view is not None:
            self.drone_view.set_velocities(physx_velocity_cmds)
        
        # 1. Force distances and headings to flat 1D views to ensure mathematical subtraction doesn't broadcast
        distances = torch.norm(self.target_pos - self.drone_pos, dim=-1).view(-1)
        headings = torch.atan2(self.target_pos[:, 1] - self.drone_pos[:, 1], self.target_pos[:, 0] - self.drone_pos[:, 0]).view(-1)
        
        # 2. Compute rewards explicitly as a 1D vector (2048,)
        rewards = (1.0 - torch.abs(headings) / 3.14159) - (0.5 * distances)
        rewards = rewards.view(-1) # Safety flatten
        
        # 3. Compute boolean tracking masks explicitly as flat 1D vectors
        terminated = (distances > 20.0).view(-1)
        truncated = (self.episode_steps >= 500).view(-1)
        # Combined mask for your internal vectorized reset tracking logic
        dones = terminated | truncated
        successes = (distances < 0.5) & (torch.abs(headings) < 0.2) 
        
        # The tensor reset must run globally *once* using vector indexes, 
        # not inside a per-env loop. This avoids unnecessary GPU kernel launches and ensures proper tensor updates.
        reset_env_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()

        if len(reset_env_ids) > 0:
            # Add successful episodes to tracking history window
            for idx in reset_env_ids:
                self.success_window.append(1.0 if successes[idx] else 0.0)
                self.episode_steps[idx] = 0

            reset_positions = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(len(reset_env_ids), 1)
            self.drone_pos[reset_env_ids] = reset_positions
            self.drone_vel[reset_env_ids] = 0.0
            
            max_spawn_radius = 2.0 + (self.difficulty_factor * 8.0) 
            random_offsets = (torch.rand((len(reset_env_ids), 2), device=self.device) * 2.0 - 1.0) * max_spawn_radius
            
            self.target_pos[reset_env_ids, 0] = self.drone_pos[reset_env_ids, 0] + random_offsets[:, 0] 
            self.target_pos[reset_env_ids, 1] = self.drone_pos[reset_env_ids, 1] + random_offsets[:, 1] 
            self.target_pos[reset_env_ids, 2] = 1.0 
            
            self.target_vel[reset_env_ids] = 0.0
            
            # 🔥 BUG FIX 3: Add conditional guards so testing/mock scripts won't crash before PhysX views boot up
            if self.drone_view is not None and self.target_view is not None:
                self.drone_view.set_world_poses(self.drone_pos[reset_env_ids], indices=reset_env_ids)
                self.target_view.set_world_poses(self.target_pos[reset_env_ids], indices=reset_env_ids)
        
        self._update_curriculum()
        next_obs = self.compute_observations()
        
        return (
            next_obs, 
            rewards.unsqueeze(-1), 
            terminated.unsqueeze(-1), 
            truncated.unsqueeze(-1), 
            {"difficulty": self.difficulty_factor, "success_rate": self._get_success_rate()}
        )