"""
Train a PPO agent to land the Lunar Lander.

    Environment: Gymnasium LunarLander-v3 (Box2D physics)
    Algorithm:   PPO from Stable-Baselines3, MlpPolicy (default 2x64 network)

Run:
    python train.py

Training takes a while on CPU (expect on the order of an hour for 1M
timesteps). When it finishes, the model is saved to models/ppo_lunar_lander.zip
and the mean episode reward of the last 100 episodes is printed.
"""

import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.results_plotter import ts2xy

from reward_wrapper import TimePenaltyWrapper

TOTAL_TIMESTEPS = 1_000_000  # ~1M steps; the standard budget for LunarLander
SEED = 42
LOG_DIR = "logs"
MODEL_PATH = os.path.join("models", "ppo_lunar_lander")


def make_env():
    """Build the training environment: reward wrapper + Monitor logging."""
    env = gym.make("LunarLander-v3")
    env = TimePenaltyWrapper(env, time_penalty=0.01)  # land sooner, not later
    env = Monitor(env, LOG_DIR)  # records per-episode rewards -> logs/*.monitor.csv
    return env


def print_ep_rew_mean():
    """Read the Monitor CSV and print mean reward of the last 100 episodes."""
    results = load_results(LOG_DIR)
    _timesteps, rewards = ts2xy(results, "timesteps")
    if len(rewards) == 0:
        print("No episode data found in logs/.")
        return
    recent = rewards[-100:]
    print(f"ep_rew_mean (last {len(recent)} episodes): {recent.mean():.1f}")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    env = make_env()

    model = PPO(
        "MlpPolicy",  # default network: 2 hidden layers x 64 units
        env,
        verbose=1,
        seed=SEED,
    )

    print(f"Training PPO for {TOTAL_TIMESTEPS:,} timesteps ...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    print("Training done.")

    print_ep_rew_mean()

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}.zip")

    env.close()


if __name__ == "__main__":
    main()
