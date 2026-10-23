[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Niteshydv8/train-your-ai/blob/main/rocket_lander_colab.ipynb)

# 🚀 Train Your AI — Rocket Lander

![Rocket render](rocket-render.webp)

Teach an AI to land a rocket on the moon. This project trains a **PPO**
(Proximal Policy Optimization) agent to land the Lunar Lander using
**Gymnasium** (Box2D physics) and **Stable-Baselines3**.

![AI training loop](ai-architecture.webp)

## ▶️ Try it in your browser

No setup needed — click the **Open in Colab** badge above, then in Colab:

- **Runtime → Run all** — trains the AI from scratch, top to bottom
  (watch a random crash → train → evaluate → watch it land → reward-shaping experiment).
- **Runtime → Restart and run all** — the restart button: wipes the trained
  model, videos, logs, and outputs, then starts every step over from the
  beginning.

## How it works (plain words)

- **The game:** a little rocket falls toward a landing pad between two
  flags. Each frame the AI sees 8 numbers: horizontal/vertical position,
  horizontal/vertical speed, tilt angle, spin speed, and whether each leg
  touches the ground.
- **The controls:** 4 buttons — do nothing, fire left engine, fire main
  (up) engine, fire right engine. The game runs at 50 frames per second.
- **The score:** landing softly between the flags earns +100 to +140,
  each leg touching down earns +10, and firing engines costs a little
  fuel penalty every frame. Average **200+ points = solved**.
- **The brain:** a small neural network (2 layers × 64 neurons) trained
  with PPO. It starts by flailing randomly, and over ~1 million frames it
  figures out how to slow down, straighten up, and touch down gently.
- **The twist (reward shaping):** we add a tiny per-step time penalty so
  the agent lands *promptly* instead of hovering, and the code includes a
  commented-out "broken" reward (+1 per frame for hovering) that teaches
  reward hacking, plus the proper fix — delta-based potential shaping.

## Project structure

```
rocket-lander-ai/
├── train.py            # trains PPO (~1M timesteps), prints ep_rew_mean, saves model
├── evaluate.py         # loads model, runs 10 episodes, reports mean reward
├── record.py           # saves landing videos (videos/*.mp4) via RecordVideo
├── reward_wrapper.py   # time-penalty wrapper + reward-shaping examples
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Python 3.10+ recommended.

```bash
# CPU-only PyTorch first (recommended; no GPU needed for this project)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# then the rest
pip install -r requirements.txt
```

> `gymnasium[box2d]` pulls in the Box2D physics engine that LunarLander-v3
> needs. On some systems you may need a C++ compiler for that step.

## Training

```bash
python train.py
```

This trains PPO for 1,000,000 timesteps (expect roughly an hour on CPU),
prints `ep_rew_mean` for the last 100 episodes, and saves the model to
`models/ppo_lunar_lander.zip`.

## The reward-shaping experiment

Open `reward_wrapper.py` — it is the most educational file here:

1. **TimePenaltyWrapper** (used in training): −0.01 reward per step, so
   dawdling in the air costs a little and landing quickly wins.
2. **Broken hover bonus** (commented out): +1 per frame for staying
   airborne. Looks sensible, but the agent learns to **hover forever**
   and never land — the reward stream never ends if the episode never
   ends. Classic reward hacking.
3. **DeltaShapingWrapper** (the fix): reward only *progress* toward the
   goal, computed as `gamma * potential(next) − potential(now)`. Any
   reward gained by moving closer is cancelled by moving away, so it
   cannot be farmed — and it provably doesn't change the optimal policy.

Try it: swap the wrapper in `train.py` and watch how the broken version
"scores high" while never landing.

## Evaluate & watch

```bash
python evaluate.py   # mean reward over 10 episodes; >= 200 means SOLVED
python record.py     # saves landing clips to videos/
```

## Results

![Training progression](training-progress.webp)

| Run | Timesteps | ep_rew_mean (last 100) | Eval mean (10 eps) | Solved? |
|-----|-----------|----------------------|--------------------|---------|
| baseline PPO + time penalty | 1,000,000 | _TBD_ | _TBD_ | _TBD_ |
| + delta shaping | 1,000,000 | _TBD_ | _TBD_ | _TBD_ |
| broken hover bonus (demo) | 300,000 | _TBD_ | _TBD_ | _TBD_ (hovers, never lands) |

_Fill these in after your runs. Training is stochastic — your numbers will
differ run to run._

## Credits

**Author:** Ram Krishna Yadav (Nitesh) — [@Niteshydv8](https://github.com/Niteshydv8)

Built with [Gymnasium](https://gymnasium.farama.org/) and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/). The training
approach here was inspired by a reinforcement-learning tutorial guide.

## License

MIT — see [LICENSE](LICENSE). Free to use, remix, and share.
