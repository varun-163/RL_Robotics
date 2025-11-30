"""PPO training on DeepMind Control Suite cartpole-swingup task.

Trains agent with multiple seeds, evaluates performance, and generates
learning curves and evaluation plots.

Prerequisites: pip install "gymnasium[other]" stable-baselines3 "shimmy>=2.0" dm-control matplotlib pandas
"""

import os
from typing import List, Tuple, Optional

import gymnasium as gym
import shimmy  # noqa: F401  # registers dm_control envs with Gymnasium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed


ENV_ID: str = "dm_control/cartpole-swingup"
TOTAL_TIMESTEPS: int = 100_000

TRAIN_SEEDS: List[int] = [0, 1, 2]
EVAL_SEED: int = 10

BASE_DIR = "./rl_results"
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

for d in [BASE_DIR, LOG_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


def make_env(seed: int, log_path: Optional[str] = None):
    """Create DMC cartpole-swingup environment, optionally with Monitor logging."""
    env = gym.make(ENV_ID)
    if log_path is not None:
        env = Monitor(env, log_path)
    env.reset(seed=seed)
    return env


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Apply moving average to smooth learning curves."""
    if window <= 1:
        return values
    if values.shape[0] < window:
        return values
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def train_single_seed(seed: int) -> str:
    """Train PPO on a single seed and return model save path."""
    print(f"=== Training (seed={seed}) ===")
    set_random_seed(seed)

    log_file = os.path.join(LOG_DIR, f"seed_{seed}")
    env = make_env(seed, log_file)

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        seed=seed,
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    save_path = os.path.join(MODELS_DIR, f"ppo_cartpole_seed_{seed}")
    model.save(save_path)
    env.close()

    print(f"[seed={seed}] Training complete. Model saved to {save_path}.zip\n")
    return save_path


def train_all_seeds():
    for seed in TRAIN_SEEDS:
        train_single_seed(seed)


def evaluate_trained_models(
    eval_seed: int = EVAL_SEED,
    n_eval_episodes: int = 10,
) -> Tuple[pd.DataFrame, float, float]:
    """Evaluate trained policies on fixed seed.
    
    Returns: (eval_df, overall_mean, overall_std)
    """
    print(f"=== Evaluation (eval_seed={eval_seed}) ===")
    eval_env = make_env(eval_seed, log_path=None)

    rows = []
    for seed in TRAIN_SEEDS:
        model_path = os.path.join(MODELS_DIR, f"ppo_cartpole_seed_{seed}")
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(
                f"Model file not found for seed {seed}: {model_path}.zip. "
                "Make sure training finished successfully."
            )

        model = PPO.load(model_path)
        mean_r, std_r = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )

        rows.append(
            {
                "train_seed": seed,
                "eval_seed": eval_seed,
                "mean_reward": mean_r,
                "std_reward": std_r,
                "n_eval_episodes": n_eval_episodes,
            }
        )
        print(f"[Eval] train_seed={seed}: mean={mean_r:.2f} ± {std_r:.2f}")

    eval_env.close()

    eval_df = pd.DataFrame(rows)
    eval_csv_path = os.path.join(BASE_DIR, "eval_results.csv")
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"Per-seed evaluation results saved to {eval_csv_path}")

    overall_mean = eval_df["mean_reward"].mean()
    overall_std = eval_df["mean_reward"].std()

    print(
        f"Final evaluation (averaged over seeds {TRAIN_SEEDS}) on eval_seed={eval_seed}: "
        f"{overall_mean:.2f} ± {overall_std:.2f}"
    )

    return eval_df, overall_mean, overall_std


def load_training_curves(
    window: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load and smooth training curves from Monitor CSV files.
    
    Returns: (steps_per_seed, rewards_per_seed)
    """
    steps_per_seed: List[np.ndarray] = []
    rewards_per_seed: List[np.ndarray] = []

    for seed in TRAIN_SEEDS:
        csv_path = os.path.join(LOG_DIR, f"seed_{seed}.monitor.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: Monitor CSV not found for seed {seed} at {csv_path}")
            continue

        df = pd.read_csv(csv_path, skiprows=1)
        steps = df["l"].cumsum().to_numpy()
        rewards = df["r"].to_numpy()

        rewards_sm = moving_average(rewards, window=window)
        if rewards_sm.shape[0] < rewards.shape[0]:
            steps_sm = steps[-rewards_sm.shape[0] :]
        else:
            steps_sm = steps

        steps_per_seed.append(steps_sm)
        rewards_per_seed.append(rewards_sm)

    if not rewards_per_seed:
        raise RuntimeError(
            "No training curves were loaded. "
            "Check that training completed and Monitor logs are present."
        )

    return steps_per_seed, rewards_per_seed


def aggregate_mean_std(
    xs: List[np.ndarray],
    ys: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncate curves to a common length and compute mean ± std across seeds."""
    min_len = min(arr.shape[0] for arr in ys)
    truncated_xs = [x[:min_len] for x in xs]
    truncated_ys = [y[:min_len] for y in ys]

    x_common = truncated_xs[0]
    y_stack = np.stack(truncated_ys, axis=0)
    mean_y = y_stack.mean(axis=0)
    std_y = y_stack.std(axis=0)
    return x_common, mean_y, std_y


def plot_learning_curve(
    steps_per_seed: List[np.ndarray],
    rewards_per_seed: List[np.ndarray],
    final_eval_mean: float,
    final_eval_std: float,
):
    """Plot learning curve with training mean±std and evaluation band."""
    x, mean_r, std_r = aggregate_mean_std(steps_per_seed, rewards_per_seed)

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_r, label="Training mean return")
    plt.fill_between(
        x,
        mean_r - std_r,
        mean_r + std_r,
        alpha=0.25,
        label="Training ±1 std (seeds 0,1,2)",
    )

    plt.axhline(
        y=final_eval_mean,
        linestyle="--",
        linewidth=2,
        label=f"Evaluation mean (eval seed = {EVAL_SEED})",
    )
    plt.fill_between(
        x,
        final_eval_mean - final_eval_std,
        final_eval_mean + final_eval_std,
        alpha=0.15,
        label="Evaluation ±1 std (across seeds)",
    )

    plt.xlabel("Environment steps (cumulative)")
    plt.ylabel("Episode return")
    plt.title("PPO on DMC Cartpole Swingup: Learning Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "learning_curve.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Main learning curve figure saved to {plot_path}")
    plt.show()


def plot_eval_summary(eval_df: pd.DataFrame):
    """Plot per-seed evaluation performance with error bars."""
    plt.figure(figsize=(6, 4))

    x_idx = np.arange(len(eval_df))
    means = eval_df["mean_reward"].to_numpy()
    stds = eval_df["std_reward"].to_numpy()
    seeds = eval_df["train_seed"].to_numpy()

    plt.bar(x_idx, means, yerr=stds, capsize=5, label="Per-seed evaluation")

    plt.xticks(x_idx, [str(s) for s in seeds])
    plt.xlabel("Training seed")
    plt.ylabel("Evaluation return (mean ± std)")
    plt.title(f"Evaluation on Cartpole Swingup (eval seed = {EVAL_SEED})")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()

    summary_path = os.path.join(PLOTS_DIR, "evaluation_summary.png")
    plt.savefig(summary_path, dpi=150)
    print(f"Evaluation summary figure saved to {summary_path}")
    plt.show()


if __name__ == "__main__":
    abs_base = os.path.abspath(BASE_DIR)
    print(f"Results will be stored under: {abs_base}\n")

    train_all_seeds()
    eval_df, final_eval_mean, final_eval_std = evaluate_trained_models()
    steps_per_seed, rewards_per_seed = load_training_curves(window=10)
    
    plot_learning_curve(
        steps_per_seed,
        rewards_per_seed,
        final_eval_mean,
        final_eval_std,
    )
    plot_eval_summary(eval_df)

    print("Done.")
