import time
from pathlib import Path
from typing import Any, Callable

from jax import Array
from jax import numpy as jnp
from jax import random as jrnd


class KArmedBandit:
    """K Armed Bandit implementation."""

    def __init__(
        self,
        n_actions: int,
        init_action_val_bias: float = 0.0,
        random_seed: int | None = None,
    ):
        """Initializes the K Armed Bandit environment.

        Args:
            n_actions (int): Number of actions in the bandit, i.e. the number of arms or "K".
            init_action_val_bias (float, optional): Bias to add to the initial action values. Defaults to 0.0.
            random_seed (int | None, optional): Random seed for reproducibility. Defaults to None.
                If None is provided, the current time is used as the seed.
        """
        random_seed = random_seed or int(time.time())
        self._random_key = jrnd.PRNGKey(random_seed)

        self._init_function = self._create_init_function(
            n_actions, init_action_val_bias
        )

        self._q_star: Array = ...
        self._Q_est: Array = ...
        self._N: Array = ...

        self.reinit()

    @property
    def optimal_action(self) -> Array:
        return jnp.argmax(self._q_star)

    def _update_random_key_decorator(func: Callable) -> Callable:
        def wrapper(self: "KArmedBandit", *args, **kwargs):
            self._update_random_key()
            return func(self, *args, **kwargs)

        return wrapper

    def _update_random_key(self) -> None:
        self._random_key = jrnd.split(self._random_key)[0]

    def _create_init_function(
        self, n_actions: int, init_action_val_bias: Array
    ) -> Callable[[], None]:

        def init_state() -> None:
            self._q_star = jrnd.normal(self._random_key, (n_actions,))
            self._Q_est = jnp.zeros(n_actions) + init_action_val_bias
            self._N = jnp.zeros(n_actions)

        return init_state

    @_update_random_key_decorator
    def reinit(self) -> None:
        self._init_function()

    @_update_random_key_decorator
    def _sample_action(self) -> Array:
        max_value = jnp.max(self._Q_est)
        tied_indices = jnp.where(self._Q_est == max_value)[0]
        selected_action_index = jrnd.choice(self._random_key, tied_indices)

        self._N = self._increment_array_at_index(
            self._N, selected_action_index, amount=1
        )

        return selected_action_index

    @_update_random_key_decorator
    def _sample_reward(self, action: int) -> Array:
        reward_mean = self._q_star[action]
        reward_std = 1.0
        return jrnd.normal(self._random_key, ()) * reward_std + reward_mean

    def _increment_array_at_index(self, arr: Array, index: int, amount: Any) -> Array:
        return arr.at[index].set(arr[index] + amount)

    def _update_action_value_estimates(self, reward: Array, action: Array) -> None:
        """Updates the action value estaimates according to the equation:
        Q(a_t) <- Q(a_t) + 1/N(a_t) * [R_t - Q(a_t)].
        """
        error = reward - self._Q_est[action]
        error = error / self._N[action]

        self._Q_est = self._increment_array_at_index(self._Q_est, action, error)

    Reward = Array
    Action = Array

    def pull(self) -> tuple[Reward, Action]:
        """Selects an action, samples a reward, and updates the action value estimates.
        Effectively pulling one arm of the bandit.

        Returns:
            tuple[Array, Array]: The reward and the action that was selected.
        """
        action = self._sample_action()
        reward = self._sample_reward(action)
        self._update_action_value_estimates(reward, action)

        return reward, action

    def save(self, destination: str) -> None:
        dest_path = Path(destination).resolve()
        jnp.savez(
            dest_path,
            q_star=self._q_star,
            Q_est=self._Q_est,
            N=self._N,
            random_key=self._random_key,
        )
