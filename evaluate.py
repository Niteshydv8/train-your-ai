"""
Evaluate a trained PPO Lunar Lander agent.

Loads models/ppo_lunar_lander.zip, runs N_EPISODES episodes with the
deterministic policy, and reports mean +/- std reward.
LunarLander-v3 counts an average reward of >= 200 as SOLVED.

Run:
    python evaluate.py
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from reward_wrapper import TimePenaltyWrapper

MODEL_PATH = "models/ppo_lunar_lander"
N_EPISODES = 10
SOLVED_THRESHOLD = 200.0


def main():
    # IMPORTANT: use the exact same wrapper (and penalty) as training,
    # otherwise the rewards you measure are not the rewards the agent
    # was trained to maximize.
    env = gym.make("LunarLander-v3")
    env = TimePenaltyWrapper(env, time_penalty=0.01)

    model = PPO.load(MODEL_PATH, env=env)
    print(f"Loaded model from {MODEL_PATH}.zip")
    print(f"Running {N_EPISODES} evaluation episodes (deterministic policy)...")

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=N_EPISODES, deterministic=True
    )

    print(f"\nMean reward: {mean_reward:.1f} +/- {std_reward:.1f} "
          f"(over {N_EPISODES} episodes)")
    if mean_reward >= SOLVED_THRESHOLD:
        print(f"SOLVED! (>= {SOLVED_THRESHOLD:.0f} mean reward)")
    else:
        print(f"Not solved yet (needs >= {SOLVED_THRESHOLD:.0f} mean reward). "
              "Try training longer.")

    env.close()


if __name__ == "__main__":
    main()
