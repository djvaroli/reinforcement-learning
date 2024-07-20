import json
import random
import string
import time
from argparse import ArgumentParser
from pathlib import Path

import seaborn as sns
from bandit import KArmedBandit
from jax import Array
from jax import numpy as jnp
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

sns.set_theme()


def generate_experiment_id(length: int = 8) -> str:
    """Generates a random experiment ID."""
    exp_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"experiment-{int(time.time())}-{exp_id}"


def average_and_plot(
    data: Array,
    xlabel: str,
    ylabel: str,
    axis: int = 0,
    save_as: str | None = None,
    fig_size: tuple[int] = (12, 10),
    label: str | None = None,
):

    averaged_data = data.mean(axis=axis)
    data_std = data.std(axis=axis)
    n_steps = data.shape[-1]

    figure = plt.figure(figsize=fig_size)

    # plot the mean reward and a shaded region representing the standard deviation
    plt.plot(averaged_data, label=label)
    plt.fill_between(
        range(n_steps), averaged_data - data_std, averaged_data + data_std, alpha=0.2
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    # save the plot
    if save_as:
        plt.savefig(save_as)


def run_experiment(
    n_arms: int,
    n_steps: int,
    n_experiments: int,
    init_Q_est: float,
    experiment_dir: str,
    random_seed: int | None = None,
):
    bandit = KArmedBandit(n_arms, init_Q_est, random_seed)

    reward_history = jnp.zeros((n_experiments, n_steps))
    selected_action_history = jnp.zeros((n_experiments, n_steps), dtype=int)
    optimal_action_history = jnp.zeros((n_experiments,), dtype=int)

    with tqdm(range(n_experiments), total=n_experiments) as pbar:
        for experiment in pbar:
            pbar.set_description(f"Run {experiment + 1}/{n_experiments}")

            # initialize the bandit. Automatically updates the random key such that
            # the next run will have a different starting random key
            bandit.reinit()
            optimal_action_history = optimal_action_history.at[experiment].set(
                bandit.optimal_action
            )

            for t in range(n_steps):
                reward, action = bandit.pull()

                reward_history = reward_history.at[experiment, t].set(reward)
                selected_action_history = selected_action_history.at[experiment, t].set(
                    action
                )

            pbar.update(1)

    experiment_dir = Path(experiment_dir).resolve()
    average_and_plot(
        reward_history,
        "Steps",
        "Average Reward",
        label=f"Runs={n_experiments}, K={n_arms}, Q0={init_Q_est}",
        save_as=experiment_dir / "average_reward.png",
    )

    was_optimal_policy_selected = (
        selected_action_history == optimal_action_history.reshape(-1, 1)
    )
    average_and_plot(
        was_optimal_policy_selected,
        "Steps",
        "% Optimal Action",
        label=f"Runs={n_experiments}, K={n_arms}, Q0={init_Q_est}",
        save_as=experiment_dir / "optimal_action.png",
    )

    bandit.save(experiment_dir / "bandit_state.npz")

    jnp.savez(
        experiment_dir / "history_arrays.npz",
        reward_history=reward_history,
        action_history=selected_action_history,
        optimal_action_history=optimal_action_history,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--n_arms", "-k", type=int, help="Number of arms in the bandit.", required=True
    )
    parser.add_argument(
        "--n_steps",
        "-n",
        type=int,
        help="Number of steps to run the bandit for.",
        required=True,
    )
    parser.add_argument(
        "--n_runs", "-r", type=int, default=1, help="Number of runs to average over."
    )
    parser.add_argument(
        "--av_bias",
        "-b",
        type=float,
        default=0.0,
        help="Bias to add to the initial action values.",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    experiment_id = generate_experiment_id()
    experiment_dir = Path(experiment_id).resolve()
    experiment_dir.mkdir(exist_ok=True)

    run_experiment(
        args.n_arms, args.n_steps, args.n_runs, args.av_bias, experiment_dir, args.seed
    )

    # save the launch configuration
    with open(f"{experiment_dir / 'launch_config.json'}", "w") as f:
        json.dump(vars(args), f)
