"""
Record a video of the trained agent landing the rocket.

Uses Gymnasium's RecordVideo wrapper (render_mode="rgb_array", moviepy
backend) to save landing clips as .mp4 files in videos/.

Run:
    python record.py
"""

import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from reward_wrapper import TimePenaltyWrapper

MODEL_PATH = "models/ppo_lunar_lander"
VIDEO_DIR = "videos"
N_EPISODES = 3


def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = TimePenaltyWrapper(env, time_penalty=0.01)
    # Record every episode we roll out below.
    env = RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda episode_id: True,
        name_prefix="lunar-lander",
    )

    model = PPO.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}.zip -- recording {N_EPISODES} episodes...")

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: reward = {total_reward:.1f}")

    env.close()  # flushes the video files to disk
    print(f"Videos saved in {VIDEO_DIR}/ (requires moviepy installed)")


if __name__ == "__main__":
    main()
