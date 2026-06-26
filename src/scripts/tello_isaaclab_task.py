# /home/dlyerly/SecurityDrone/src/scripts/tello_isaaclab_task.py
#
# Real Isaac Lab DirectRLEnv task definition for "TelloAdaptiveTracking".
# This module is only importable on a real Isaac Sim / Omniverse install (AWS box) --
# it is intentionally kept out of tello_env_cfg.py so local mock training still works
# on machines without isaaclab installed. Import this module once, before calling
# load_isaaclab_env(), to register the task with gymnasium.
from __future__ import annotations

import os
from collections import deque

import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Produced by convert_tello.py (urdf -> usd). Override with TELLO_USD_PATH if you
# convert/store it elsewhere on the AWS instance.
TELLO_USD_PATH = os.environ.get(
    "TELLO_USD_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "tello.usd"),
)

TELLO_ROBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TELLO_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
)


@configclass
class TelloAdaptiveEnvCfg(DirectRLEnvCfg):
    # one episode == 500 sim steps at 60Hz, matching the original mock's truncation
    decimation = 2
    episode_length_s = 500 * (1 / 60) * decimation
    action_space = 4  # [roll, pitch, throttle, yaw] mapped to thrust + body moments
    observation_space = 6  # [rel_pos(3), rel_vel(3)], matching the original mock
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain: TerrainImporterCfg = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)
    robot: ArticulationCfg = TELLO_ROBOT_CFG

    thrust_to_weight = 3.0
    moment_scale = 0.01
    max_distance = 20.0
    success_distance = 0.5
    success_heading = 0.2
    spawn_radius_min = 2.0
    spawn_radius_max = 10.0
    curriculum_up_threshold = 0.85
    curriculum_down_threshold = 0.50
    curriculum_step = 0.05
    obs_noise_scale = 0.15


class TelloAdaptiveEnv(DirectRLEnv):
    cfg: TelloAdaptiveEnvCfg

    def __init__(self, cfg: TelloAdaptiveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self._difficulty_factor = 0.0
        self._success_window: deque[float] = deque(maxlen=100)

        self._body_id = self._robot.find_bodies("base_link")[0]
        robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        gravity_magnitude = abs(self.sim.cfg.gravity[2])
        self._robot_weight = (robot_mass * gravity_magnitude).item()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        roll, pitch, throttle, yaw = (
            self._actions[:, 0],
            self._actions[:, 1],
            self._actions[:, 2],
            self._actions[:, 3],
        )

        self._thrust[:, 0, 2] = self._robot_weight * (throttle + 1.0) / 2.0 * self.cfg.thrust_to_weight
        self._moment[:, 0, 0] = pitch * self.cfg.moment_scale
        self._moment[:, 0, 1] = roll * self.cfg.moment_scale
        self._moment[:, 0, 2] = yaw * self.cfg.moment_scale

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _rel_pos_vel(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos = self._robot.data.root_pos_w - self.scene.env_origins
        root_vel = self._robot.data.root_lin_vel_w
        return self._target_pos - root_pos, root_vel

    def _get_observations(self) -> dict:
        rel_pos, root_vel = self._rel_pos_vel()
        noise_amplitude = self.cfg.obs_noise_scale * self._difficulty_factor
        noisy_rel_pos = rel_pos + torch.randn_like(rel_pos) * noise_amplitude
        noisy_vel = root_vel + torch.randn_like(root_vel) * noise_amplitude
        return {"policy": torch.cat([noisy_rel_pos, noisy_vel], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        rel_pos, _ = self._rel_pos_vel()
        distances = torch.norm(rel_pos, dim=-1)
        headings = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
        return (1.0 - torch.abs(headings) / 3.14159) - (0.5 * distances)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        rel_pos, _ = self._rel_pos_vel()
        distances = torch.norm(rel_pos, dim=-1)
        terminated = distances > self.cfg.max_distance
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _update_curriculum(self):
        if not self._success_window:
            return
        success_rate = sum(self._success_window) / len(self._success_window)
        if success_rate > self.cfg.curriculum_up_threshold and self._difficulty_factor < 1.0:
            self._difficulty_factor = min(1.0, self._difficulty_factor + self.cfg.curriculum_step)
        elif success_rate < self.cfg.curriculum_down_threshold and self._difficulty_factor > 0.0:
            self._difficulty_factor = max(0.0, self._difficulty_factor - self.cfg.curriculum_step)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        rel_pos = self._target_pos[env_ids] - (self._robot.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids])
        distances = torch.norm(rel_pos, dim=-1)
        headings = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
        successes = (distances < self.cfg.success_distance) & (torch.abs(headings) < self.cfg.success_heading)
        for success in successes.tolist():
            self._success_window.append(1.0 if success else 0.0)
        self._update_curriculum()

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        max_spawn_radius = self.cfg.spawn_radius_min + self._difficulty_factor * (
            self.cfg.spawn_radius_max - self.cfg.spawn_radius_min
        )
        offsets = (torch.rand(len(env_ids), 2, device=self.device) * 2.0 - 1.0) * max_spawn_radius
        self._target_pos[env_ids, 0] = offsets[:, 0]
        self._target_pos[env_ids, 1] = offsets[:, 1]
        self._target_pos[env_ids, 2] = 1.0


gym.register(
    id="TelloAdaptiveTracking",
    entry_point=f"{__name__}:TelloAdaptiveEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:TelloAdaptiveEnvCfg"},
)
