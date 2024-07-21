import json
import random
import string
import time
from argparse import ArgumentParser
from pathlib import Path

import seaborn as sns
from action.selection import EpsilonGreedyActionSelection, GreeedyActionSelection
from bandit import KArmedBandit
from jax import Array
from jax import numpy as jnp
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

sns.set_theme()


class ExperimentDataContainer:
    def __init__(
        self,
        n_experiments: int,
        n_steps: int,
    ):
        self.reward_history = jnp.zeros((n_experiments, n_steps))
        self.selected_action_history = jnp.zeros((n_experiments, n_steps), dtype=int)
        self.optimal_action_history = jnp.zeros((n_experiments,), dtype=int)

    @property
    def n_experiments(self) -> int:
        return self.reward_history.shape[0]

    @property
    def n_steps(self) -> int:
        return self.reward_history.shape[1]

    def update_rewards(self, experiment: int, step: int, reward: Array) -> None:
        self.reward_history = self.reward_history.at[experiment, step].set(reward)

    def update_selected_actions(self, experiment: int, step: int, action: int) -> None:
        self.selected_action_history = self.selected_action_history.at[
            experiment, step
        ].set(action)

    def update_optimal_action(self, experiment: int, optimal_action: int) -> None:
        self.optimal_action_history = self.optimal_action_history.at[experiment].set(
            optimal_action
        )

    def save(self, path: str) -> None:
        jnp.savez(
            path,
            reward_history=self.reward_history,
            selected_action_history=self.selected_action_history,
            optimal_action_history=self.optimal_action_history,
        )


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
    bandit: KArmedBandit,
    n_experiments: int,
    n_steps: int,
) -> ExperimentDataContainer:
    experiment_data = ExperimentDataContainer(n_experiments, n_steps)

    with tqdm(range(n_experiments), total=n_experiments) as pbar:
        for experiment in range(n_experiments):
            pbar.set_description(f"Run {experiment + 1}/{n_experiments}")

            # re-initialize bandit for next experiment run. Automatically updates random state
            bandit.reinit()
            experiment_data.update_optimal_action(experiment, bandit.optimal_action)

            for t in range(n_steps):
                reward, action = bandit.pull()
                experiment_data.update_rewards(experiment, t, reward)
                experiment_data.update_selected_actions(experiment, t, action)

            pbar.update(1)

    return experiment_data


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--n_arms", "-k", type=int, help="Number of arms in the bandit.", required=True
    )
    parser.add_argument(
        "--action_selector",
        "-a",
        type=str,
        help="Action selection method to use.",
        required=True,
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
        "--action_value_bias",
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

    action_selection_method = eval(args.action_selector)
    print(action_selection_method)

    bandit = KArmedBandit(
        args.n_arms, action_selection_method, args.action_value_bias, args.seed
    )

    start_time = time.perf_counter()
    experiment_data = run_experiment(
        bandit,
        args.n_runs,
        args.n_steps,
    )

    end_time = time.perf_counter()

    label = ""
    for key, value in vars(args).items():
        label += f"{key}={value}, "

    average_and_plot(
        experiment_data.reward_history,
        "Steps",
        "Average Reward",
        label=label,
        save_as=experiment_dir / "average_reward.png",
    )

    was_optimal_policy_selected = (
        experiment_data.selected_action_history
        == experiment_data.optimal_action_history.reshape(-1, 1)
    )

    average_and_plot(
        was_optimal_policy_selected,
        "Steps",
        "% Optimal Action",
        label=label,
        save_as=experiment_dir / "optimal_action.png",
    )

    bandit.save(experiment_dir / "bandit_state.npz")
    experiment_data.save(experiment_dir / "experiment_data.npz")

    config = vars(args)
    config["experiment_id"] = experiment_id
    config["execution_time"] = int(end_time - start_time)

    # save the launch configuration
    with open(f"{experiment_dir / 'launch_config.json'}", "w") as f:
        json.dump(config, f)
