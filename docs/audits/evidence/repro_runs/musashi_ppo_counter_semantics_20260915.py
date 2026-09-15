"""CPU-only library unit probe; not a scientific campaign or trading run."""
import json
import gymnasium as gym
import stable_baselines3
import torch
from stable_baselines3 import PPO


def main():
    torch.set_num_threads(1)
    env = gym.make("CartPole-v1")
    try:
        model = PPO("MlpPolicy", env, n_steps=256, batch_size=64,
                    n_epochs=10, device="cpu", seed=0, verbose=0)
        calls = 0
        original = model.policy.optimizer.step

        def counted_step(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            return result

        model.policy.optimizer.step = counted_step
        model.learn(64)
        result = {"sb3_version": stable_baselines3.__version__, "requested": 64,
                  "observed_env_steps": model.num_timesteps,
                  "sb3_n_updates": model._n_updates, "optimizer_step_calls": calls}
        print(json.dumps(result, sort_keys=True))
        assert (model.num_timesteps, model._n_updates, calls) == (256, 10, 40)
    finally:
        env.close()


if __name__ == "__main__":
    main()
