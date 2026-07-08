# /home/dlyerly/SecurityDrone/src/configs/tello_curriculum.py
import torch

class TelloCurriculumManager:
    @staticmethod
    def apply_domain_randomization(env):
        """
        Applies multi-stage curriculum domain randomization for the low-level policy.
        Scales observation noise, action history latency, and aerodynamic disturbances.
        """
        df = env.difficulty_factor
        
        # 1. State Estimation & IMU Noise Scale
        env.pos_noise_sigma = 0.05 * df
        env.vel_noise_sigma = 0.08 * df
        
        # 2. Camera Sensor Glare / Dropout Rate
        env.depth_dropout_prob = 0.15 * df
        
        # 3. Dynamic Action History Latency Limits
        # Phase 0: No lag. Phase 3: Up to 4 steps of execution delay.
        env.current_max_latency = int(1 + (df * 3)) 
        
        # 4. Aerodynamic Disturbance (Wind Gusts / Ground Effect)
        env.wind_force_magnitude = 2.5 * df  # Newtons of peak sudden force
        env.wind_probability = 0.05 * df      # Chance of a gust per timestep

        
    @staticmethod
    def compute_reward(difficulty_factor, drone_pos, target_pos, drone_vel, actions, headings, distances, terminated):
        """
        Dynamically adjusts and computes the reward functions based on the current curriculum tier.

        Args:
            difficulty_factor (float): Progress score between 0.0 and 1.0.
            drone_pos, target_pos (Tensor): Spatial 3D positions [num_envs, 3].
            drone_vel (Tensor): Spatial 3D linear velocity [num_envs, 3].
            actions (Tensor): Raw policy network actions clamped between [-1, 1].
            headings (Tensor): Flat 1D orientation tracking error.
            distances (Tensor): Flat 1D Euclidean spatial separation vector.
            terminated (Tensor): Flat 1D Boolean boundary breach state flag.
        """
        # --- LEVEL 0: Early Phase (Static/Slow Target, Basic Proximity Orientation) ---
        if difficulty_factor <= 0.2:
            r_heading = (1.0 - torch.abs(headings) / 3.14159)
            r_dist = -0.5 * distances
            rewards = r_heading + r_dist

        # --- LEVEL 1: Intermediate Phase (Introduce Action Smoothness Regularization) ---
        elif difficulty_factor <= 0.5:
            r_heading = (1.0 - torch.abs(headings) / 3.14159)
            r_dist = -0.4 * distances
            r_action = -0.01 * torch.sum(torch.square(actions), dim=-1)
            rewards = r_heading + r_dist + r_action

        # --- LEVEL 2: Advanced Phase (Exponential Clamping & Kinetic Penalties) ---
        elif difficulty_factor <= 0.8:
            r_heading = torch.exp(-1.0 * torch.abs(headings))
            r_dist = torch.exp(-0.5 * distances)
            r_action = -0.02 * torch.sum(torch.square(actions), dim=-1)
            r_vel = -0.05 * torch.norm(drone_vel, dim=-1)  # Penalize heavy oscillations/overshooting
            rewards = r_heading + r_dist + r_action + r_vel

        # --- LEVEL 3: Expert Phase (Tight Target Bounds & High-Speed Flight Stability) ---
        else:
            r_heading = torch.exp(-2.0 * torch.abs(headings))
            r_dist = torch.exp(-1.0 * distances)
            r_action = -0.05 * torch.sum(torch.square(actions), dim=-1)
            r_vel = -0.10 * torch.norm(drone_vel, dim=-1)
            rewards = r_heading + r_dist + r_action + r_vel

        # --- GLOBAL CONSTRAINTS: Crash & Out-of-Bounds Terminations ---
        rewards = torch.where(terminated, rewards - 15.0, rewards)

        return rewards.view(-1)