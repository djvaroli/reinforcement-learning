from jax import Array
from jax import numpy as jnp
from jax import random as jrnd

from .base import ActionSelection


class GreeedyActionSelection(ActionSelection):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def select(self, random_key: Array, est_action_values: Array) -> Array:
        val_max = jnp.max(est_action_values)
        ties = jnp.where(est_action_values == val_max)[0]
        return jrnd.choice(random_key, ties)


class EpsilonGreedyActionSelection(ActionSelection):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(epsilon={self.epsilon})"

    def __init__(self, epsilon: float):
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1.")

        self.epsilon = epsilon

    def select(self, random_key: Array, est_action_values: Array) -> Array:
        # sample from a uniform distribution
        if jrnd.uniform(random_key, (1,)) < self.epsilon:
            return jrnd.choice(random_key, est_action_values)

        return super().select(random_key, est_action_values)
