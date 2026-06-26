# /home/dlyerly/SecurityDrone/src/scripts/smoke_test_isaaclab.py
#
# Cheap sanity check for the "TelloAdaptiveTracking" Isaac Lab task, meant to run
# BEFORE train.py's full 10M-timestep job. Uses a handful of envs and a handful of
# steps so it costs seconds of AWS time instead of a full billed training run.
#
# Run on the AWS Isaac Sim instance:
#   python smoke_test_isaaclab.py --headless --num_envs 4 --steps 5
import argparse

import tello_isaaclab_task  # noqa: F401  (registers "TelloAdaptiveTracking")
from skrl.envs.loaders.torch import load_isaaclab_env


def main():
    parser = argparse.ArgumentParser(description="Smoke test for TelloAdaptiveTracking")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    print(f"[SMOKE TEST] Loading TelloAdaptiveTracking with num_envs={args.num_envs}...")
    env = load_isaaclab_env(task_name="TelloAdaptiveTracking", num_envs=args.num_envs, headless=args.headless)

    obs, info = env.reset()
    print(f"[SMOKE TEST] reset OK. obs type={type(obs)}")
    _print_shape("obs", obs)

    expected_obs_dim = 6
    _assert_last_dim(obs, expected_obs_dim, "observation")

    for i in range(args.steps):
        actions = _random_actions(env)
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"[SMOKE TEST] step {i}: rewards shape={tuple(rewards.shape)} "
              f"terminated shape={tuple(terminated.shape)} truncated shape={tuple(truncated.shape)}")
        _assert_last_dim(obs, expected_obs_dim, "observation")
        assert rewards.shape[0] == args.num_envs, f"reward batch dim mismatch: {rewards.shape}"

    env.close()
    print("[SMOKE TEST] PASSED -- shapes look correct. Safe to consider the full training run.")


def _random_actions(env):
    import torch
    return torch.rand(env.num_envs, env.action_space.shape[0], device=env.device) * 2.0 - 1.0


def _print_shape(name, value):
    if hasattr(value, "shape"):
        print(f"[SMOKE TEST] {name} shape={tuple(value.shape)}")
    elif isinstance(value, dict):
        for k, v in value.items():
            _print_shape(f"{name}.{k}", v)


def _assert_last_dim(value, expected, label):
    if hasattr(value, "shape"):
        assert value.shape[-1] == expected, f"{label} last dim {value.shape[-1]} != expected {expected}"
    elif isinstance(value, dict):
        for v in value.values():
            _assert_last_dim(v, expected, label)


if __name__ == "__main__":
    main()
