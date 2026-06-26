# /home/dlyerly/SecurityDrone/src/scripts/train.py
import os
import argparse
import yaml
import torch
import torch.nn as nn

# 1. Native Isaac Lab & SKRL Production imports
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.utils.runner.torch import Runner
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

# Dynamic environment routing wrapper guard
try:
    import omni # Looks for the real simulator footprint
    from skrl.envs.loaders.torch import load_isaaclab_env
    from skrl.envs.wrappers.torch import wrap_env
    AWS_MODE = True
except ImportError:
    # Safe fallback for your local development sandbox
    from local_mock.env_wrapper import SkrlVecEnvWrapper
    AWS_MODE = False

# --- Framework-Compliant Model Implementations ---
class TelloPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, backbone):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, role="policy")
        nn.Module.__init__(self)
        
        self.backbone = backbone
        self.actor_head = nn.Linear(64, action_space.shape[0])
        self.log_std_parameter = nn.Parameter(torch.zeros((action_space.shape[0],), device=device))

    def compute(self, inputs, role):
        states = inputs.get("states", None)
        if states is None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and len(value.shape) > 1 and value.shape[-1] == 6:
                    states = value
                    break
        features = self.backbone(states)
        mean_actions = self.actor_head(features)
        return mean_actions, {"log_std": self.log_std_parameter}

class TelloValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, backbone):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, role="value")
        nn.Module.__init__(self)
        
        self.backbone = backbone
        self.critic_head = nn.Linear(64, 1)

    def compute(self, inputs, role):
        states = inputs.get("states", None)
        if states is None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and len(value.shape) > 1 and value.shape[-1] == 6:
                    states = value
                    break
        features = self.backbone(states)
        return self.critic_head(features), {}

class SharedBackbone(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )
    def forward(self, states):
        return self.net(states)

# --- Execution Entry Point ---
def main():
    # 1. Parse arguments and dynamically check for a local GPU footprint
    parser = argparse.ArgumentParser(description="Tello Drone RL Training Loop")
    parser.add_argument("--headless", action="store_true", help="Run simulation headlessly")
    
    # Dynamically fall back to CPU if your local machine lacks an NVIDIA driver
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device, help="Target computing hardware")
    
    args = parser.parse_args()
    print(f"[INFO] Using target execution device: {args.device}")

    # 2. Dynamic Environment Routing based on your machine type
    if AWS_MODE:
        print("[INFO] Launching real Isaac Sim production instance on AWS...")
        import tello_isaaclab_task  # noqa: F401  (registers "TelloAdaptiveTracking" with gymnasium)
        raw_env = load_isaaclab_env(task_name="TelloAdaptiveTracking", headless=args.headless)
        env = wrap_env(raw_env)
    else:
        print("[INFO] Running in local mock sandbox environment context...")
        from tello_env_cfg import TelloAdaptiveEnv 
        raw_env = TelloAdaptiveEnv(num_envs=2048, device=args.device)
        env = SkrlVecEnvWrapper(raw_env)

    # 3. Load production parameters from configuration file
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "tello_ppo_cfg.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- DUPLICATE RA_ENV / ENV BLOCKS REMOVED FROM HERE ---

    # 4. Initialize shared model representations using the dynamically selected 'env'
    backbone = SharedBackbone(env.observation_space.shape[0]).to(args.device)
    models = {
        "policy": TelloPolicy(env.observation_space, env.action_space, args.device, backbone),
        "value": TelloValue(env.observation_space, env.action_space, args.device, backbone)
    }

    print("[INFO] Instantiating production SKRL execution structures...")
    # Initialize rollout tracking buffer
    memory = RandomMemory(
        memory_size=cfg["memory"]["memory_size"], 
        num_envs=env.num_envs, 
        device=args.device
    )

    # Map device parameters into configuration blocks
    from skrl.agents.torch.ppo import PPO_CFG
    agent_cfg = PPO_CFG(**cfg["agent"])
    
    #  Set the device directly as an attribute on the config object
    agent_cfg.device = args.device

    # Initialize agent logic
    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space
    )

    # 1. Import the explicit loop trainer at the top of your script or right here
    from skrl.trainers.torch import SequentialTrainer

    # 2. Initialize the trainer configuration
    trainer_cfg = {
        "timesteps": cfg["trainer"]["timesteps"],
        "close_environment_at_exit": cfg["trainer"]["close_environment_at_exit"]
    }
    
    # 3. Instantiate the explicit trainer engine
    trainer = SequentialTrainer(
        env=env, 
        agents=[agent],  # Passes your pre-built PPO agent directly
        cfg=trainer_cfg
    )

    print("[INFO] Handing control over to the SKRL sequential loop engine...")
    trainer.train()

if __name__ == "__main__":
    main()