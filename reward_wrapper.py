"""
Reward shaping for LunarLander-v3.

What the environment already rewards (built-in):
    +100 to +140  for touching down between the landing flags
    +10           for each leg touching the ground
    -0.3          per frame while the main engine fires
    -0.03         per frame while a side engine fires
An episode is "solved" at an average reward of 200+.

This file contains:
    1. TimePenaltyWrapper -- the wrapper actually used in train.py.
       Adds a tiny per-step penalty so the agent prefers landing
       sooner rather than dawdling in the air.
    2. A BROKEN reward idea (commented out): +1 per frame for hovering.
       It looks helpful but teaches the agent to hover forever instead
       of landing. A textbook case of reward hacking.
    3. DeltaShapingWrapper -- the fixed version: reward only *progress*
       toward the goal using potential-based shaping, which cannot be
       farmed the way the hover bonus can.
"""

import gymnasium as gym


class TimePenaltyWrapper(gym.Wrapper):
    """
    Adds a small constant penalty to every step's reward.

    Why: without it, hovering costs the agent almost nothing, so it can
    learn to waste time in the air. A tiny -0.01 per step makes dawdling
    slightly painful, so landing quickly becomes the better deal.
    The penalty is small enough that it never drowns out the real
    landing reward (+100..140).
    """

    def __init__(self, env, time_penalty=0.01):
        super().__init__(env)
        self.time_penalty = time_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward - self.time_penalty
        return obs, reward, terminated, truncated, info


class DeltaShapingWrapper(gym.Wrapper):
    """
    Potential-based shaping (Ng, Harada & Russell, 1999).

    Define a "potential" P(s) that is high when the lander is in a good
    state (low, slow, upright, near the pad) and low otherwise. The
    shaping reward for a transition is:

        shaping = gamma * P(next_state) - P(current_state)

    Because the shaping is a *difference* of potentials, any reward the
    agent earns by moving toward the goal is exactly cancelled if it
    moves back away. Over a whole episode the extra reward telescopes to
    gamma * P(final) - P(start), so there is no way to farm infinite
    reward by looping in place -- unlike the broken hover bonus below.

    Bonus property: potential-based shaping provably does not change the
    optimal policy; it only helps the agent discover it faster.
    """

    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self._last_potential = None

    def _potential(self, obs):
        # obs = [x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact]
        _x, y, vx, vy, angle, ang_vel, _leg1, _leg2 = obs
        # High potential = low altitude, low speed, upright, pad-centered.
        return -(abs(y) + abs(vx) + abs(vy) + 5.0 * abs(angle) + 0.5 * abs(ang_vel))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_potential = self._potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        potential = self._potential(obs)
        shaping = self.gamma * potential - self._last_potential
        self._last_potential = potential
        return obs, reward + shaping, terminated, truncated, info


# ---------------------------------------------------------------------------
# BROKEN EXAMPLE -- DO NOT USE. Kept here so you can see the failure mode.
#
# The idea: "reward the lander for staying airborne -- hovering is skillful!"
#
#     class HoverBonusWrapper(gym.Wrapper):
#         def step(self, action):
#             obs, reward, terminated, truncated, info = self.env.step(action)
#             y = obs[1]              # altitude
#             if y > 0.5:             # still in the air
#                 reward += 1.0       # +1 every single frame for hovering
#             return obs, reward, terminated, truncated, info
#
# What the agent actually learns: NEVER LAND.
# Hovering forever earns +1/frame with zero risk, while landing ends the
# episode and stops the reward stream. The agent discovers that the best
# strategy is to burn fuel gently and float -- it racks up a huge score
# without ever touching the pad. That is reward hacking: the agent
# optimizes your reward function instead of your actual goal.
#
# Rule of thumb: never reward a *state* ("being in the air"); reward
# *progress* ("getting closer to the pad"). See DeltaShapingWrapper above.
# ---------------------------------------------------------------------------
