# /home/dlyerly/SecurityDrone/src/scripts/tello_low_level_env.py
import torch
import gymnasium as gym
from collections import deque
from configs.tello_curriculum import TelloCurriculumManager

class TelloLowLevelEnv(gym.Env):
    def __init__(self, num_envs=2048, device="cuda"):
        super().__init__()
        self.num_envs = num_envs
        self.device = device
        
        # --- DEFINITIONS ---
        # Obs: [Rel_XYZ (3), Lin_Vel (3), Ang_Vel (3), Downsampled_Depth (32)] = 41 Prev_Action (4): [v_x, v_y, v_z, w_z]
        self.observation_space_dim = 45
        # Actions: [v_x, v_y, v_z, w_z] = 4
        self.action_space_dim = 4 

        self.observation_space = gym.spaces.Box(
            low=-float('inf'), high=float('inf'), shape=(self.observation_space_dim,), dtype=float
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_space_dim,), dtype=float
        )
        
        # --- CURRICULUM & TRACKING ---
        self.difficulty_factor = 0.0  
        self.success_window = deque(maxlen=100)
        self.episode_steps = torch.zeros(self.num_envs, device=self.device)
        
        # --- STATE BUFFERS ---
        self.drone_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drone_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.drone_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # 1x32 Downsampled Depth Array Vector Placeholder
        self.depth_inputs = torch.ones((self.num_envs, 32), device=self.device) * 5.0 # Max range 5m
        
        # --- LATENCY ACTION QUEUE ---
        self.max_queue_len = 5
        # Pre-fill action history queue matrices [Max_Steps, Num_Envs, Action_Dim]
        self.action_queue = torch.zeros((self.max_queue_len, self.num_envs, self.action_space_dim), device=self.device)
        self.step_delay_per_env = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Isaac Lab View References
        self.drone_view = None 
        self.target_view = None

    def compute_observations(self):
        """Processes depth dropout, state estimation drift, and IMU noise."""
        TelloCurriculumManager.apply_domain_randomization(self)
        
        # State Estimation Drift (Relative Target Position)
        real_rel_pos = self.target_pos - self.drone_pos
        pos_noise = torch.randn_like(real_rel_pos) * self.pos_noise_sigma
        noisy_rel_pos = real_rel_pos + pos_noise
        
        # IMU Reading Noise (Linear and Angular Velocities)
        noisy_lin_vel = self.drone_vel + (torch.randn_like(self.drone_vel) * self.vel_noise_sigma)
        noisy_ang_vel = self.drone_ang_vel + (torch.randn_like(self.drone_ang_vel) * self.vel_noise_sigma)
        
        # Camera Glare/Compression Dropout Masking
        noisy_depth = self.depth_inputs.clone()
        dropout_mask = torch.rand_like(noisy_depth) < self.depth_dropout_prob
        # Randomly blind elements by forcing them to 0.0 (total glare drop)
        noisy_depth[dropout_mask] = 0.0

        # Most recently injected action (zeros on reset)
        prev_action = self.action_queue[0]

        return torch.cat([noisy_rel_pos, noisy_lin_vel, noisy_ang_vel, noisy_depth, prev_action], dim=-1)

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        
        self.episode_steps.fill_(0)
        self.drone_pos.fill_(0.0)
        self.drone_pos[:, 2] = 1.0  # Launch hover altitude
        self.drone_vel.fill_(0.0)
        self.drone_ang_vel.fill_(0.0)
        self.action_queue.fill_(0.0)
        
        # Initial step delays randomly sampled between 1 and current max latency
        TelloCurriculumManager.apply_domain_randomization(self)
        self.step_delay_per_env = torch.randint(1, self.current_max_latency + 1, (self.num_envs,), device=self.device)
        
        # Spawn targets randomly within tracking zone
        self.target_pos[:, 0] = (torch.rand(self.num_envs, device=self.device) * 2.0 - 1.0) * 5.0
        self.target_pos[:, 1] = (torch.rand(self.num_envs, device=self.device) * 2.0 - 1.0) * 5.0
        self.target_pos[:, 2] = 1.0 + (torch.rand(self.num_envs, device=self.device) * 2.0 - 1.0) * 0.5
        
        if self.drone_view is not None and self.target_view is not None:
            all_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
            self.drone_view.set_world_poses(self.drone_pos, indices=all_indices)
            self.target_view.set_world_poses(self.target_pos, indices=all_indices)
            
        return self.compute_observations(), {"difficulty": self.difficulty_factor}

    def step(self, actions):
        self.episode_steps += 1
        actions = torch.clamp(actions, min=-1.0, max=1.0)
        
        # 1. --- ACTION HISTORY LATENCY QUEUE SHIFT ---
        # Roll queue forward along step dimension
        self.action_queue = torch.roll(self.action_queue, shifts=1, dims=0)
        self.action_queue[0] = actions  # Inject current freshest step command
        
        # Extract individual delayed indexes matching each environment's latency budget
        # Using advanced indexing to pluck out specific time delays per env row
        delayed_actions = self.action_queue[self.step_delay_per_env, torch.arange(self.num_envs)]
        
        # 2. --- AERODYNAMIC DISTURBANCE (WIND INJECTIONS) ---
        if self.drone_view is not None and self.wind_probability > 0.0:
            # Sample which environment instances get smacked by air drafts this frame
            gust_mask = torch.rand(self.num_envs, device=self.device) < self.wind_probability
            if gust_mask.any():
                num_gusts = gust_mask.sum().item()
                # Apply random force vectors (X, Y, Z)scaled by curriculum peak force limits
                random_forces = (torch.rand((num_gusts, 3), device=self.device) * 2.0 - 1.0) * self.wind_force_magnitude
                # Inject directly into PhysX body indices via Isaac Sim C++ backend structures
                gust_indices = gust_mask.nonzero().flatten().to(torch.int32)
                self.drone_view.apply_forces(random_forces, indices=gust_indices)

        # Map delayed action variables to PhysX velocity commands
        physx_velocity_cmds = torch.zeros((self.num_envs, 6), device=self.device)
        physx_velocity_cmds[:, 0] = delayed_actions[:, 0] * 3.0  # Max v_x = 3.0 m/s
        physx_velocity_cmds[:, 1] = delayed_actions[:, 1] * 3.0  # Max v_y = 3.0 m/s
        physx_velocity_cmds[:, 2] = delayed_actions[:, 2] * 2.0  # Max v_z = 2.0 m/s
        physx_velocity_cmds[:, 5] = delayed_actions[:, 3] * 3.14 # Max w_z = pi rad/s
        
        if self.drone_view is not None:
            self.drone_view.set_velocities(physx_velocity_cmds)
            
        # 3. --- WAYPOINT SNAPPING PATTERN ENGINE ---
        distances = torch.norm(self.target_pos - self.drone_pos, dim=-1).view(-1)
        headings = torch.atan2(self.target_pos[:, 1] - self.drone_pos[:, 1], self.target_pos[:, 0] - self.drone_pos[:, 0]).view(-1)
        
        # Detect proximity threshold crossings [num_envs]
        snap_mask = distances < 0.5
        snap_env_ids = snap_mask.nonzero(as_tuple=False).flatten()
        
        if len(snap_env_ids) > 0:
            # Teleport targets instantly to force sharp banking redirects
            new_targets = (torch.rand((len(snap_env_ids), 3), device=self.device) * 2.0 - 1.0) * 6.0
            new_targets[:, 2] = 1.0 + (torch.rand(len(snap_env_ids), device=self.device) * 2.0 - 1.0) * 0.5
            self.target_pos[snap_env_ids] = new_targets
            
            # Re-sample latency budgets for these specific environments to keep PPO predicting
            self.step_delay_per_env[snap_env_ids] = torch.randint(1, self.current_max_latency + 1, (len(snap_env_ids),), device=self.device)
            
            if self.target_view is not None:
                self.target_view.set_world_poses(self.target_pos[snap_env_ids], indices=snap_env_ids.to(torch.int32))
                
            for idx in snap_env_ids:
                self.success_window.append(1.0) # Confirmed hit

        # Boundary rules and timeouts
        terminated = (distances > 25.0).view(-1)
        truncated = (self.episode_steps >= 750).view(-1) # Longer tracks for continuous momentum
        
        # Hard resets for out-of-bounds or terminal timeouts
        reset_env_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            for idx in reset_env_ids:
                if not snap_mask[idx]: # If it didn't snap, it failed or timed out
                    self.success_window.append(0.0)
            
            # Standard reset positions
            self.drone_pos[reset_env_ids] = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(len(reset_env_ids), 1)
            self.drone_vel[reset_env_ids] = 0.0
            self.drone_ang_vel[reset_env_ids] = 0.0
            self.episode_steps[reset_env_ids] = 0
            
            self.target_pos[reset_env_ids, 0] = (torch.rand(len(reset_env_ids), device=self.device) * 2.0 - 1.0) * 5.0
            self.target_pos[reset_env_ids, 1] = (torch.rand(len(reset_env_ids), device=self.device) * 2.0 - 1.0) * 5.0
            self.target_pos[reset_env_ids, 2] = 1.0
            
            if self.drone_view is not None and self.target_view is not None:
                self.drone_view.set_world_poses(self.drone_pos[reset_env_ids], indices=reset_env_ids.to(torch.int32))
                self.target_view.set_world_poses(self.target_pos[reset_env_ids], indices=reset_env_ids.to(torch.int32))

        # Dynamic metric updates
        success_rate = sum(self.success_window) / len(self.success_window) if self.success_window else 0.0
        if success_rate > 0.88 and self.difficulty_factor < 1.0:
            self.difficulty_factor = min(1.0, self.difficulty_factor + 0.04)
        elif success_rate < 0.45 and self.difficulty_factor > 0.0:
            self.difficulty_factor = max(0.0, self.difficulty_factor - 0.04)

        # 🔥 Call your empty custom reward function placeholder
        rewards = TelloCurriculumManager.compute_reward(
            difficulty_factor=self.difficulty_factor,
            drone_pos=self.drone_pos,
            target_pos=self.target_pos,
            drone_vel=self.drone_vel,
            actions=actions,
            headings=headings,
            distances=distances,
            terminated=terminated
        )
        next_obs = self.compute_observations()
        
        return (
            next_obs,
            rewards.unsqueeze(-1),
            terminated.unsqueeze(-1),
            truncated.unsqueeze(-1),
            {"difficulty": self.difficulty_factor, "success_rate": success_rate}
        )