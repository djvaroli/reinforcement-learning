from jax import Array
from jax import numpy as jnp
from jax import random as jrnd

from .base import ActionSelection
from .utils import _argmax_break_ties


class EpsilonGreedyActionSelection(ActionSelection):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(epsilon={self.epsilon})"

    def __init__(self, epsilon: float):
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1.")

        self.epsilon = epsilon

    def select(
        self, random_key: Array, est_action_values: Array, selection_count: Array
    ) -> Array:
        # sample from a uniform distribution
        if jrnd.uniform(random_key, (1,)) < self.epsilon:
            indices = jnp.arange(est_action_values.shape[0])
            return jrnd.choice(random_key, indices)

        return _argmax_break_ties(random_key, est_action_values)


class GreeedyActionSelection(EpsilonGreedyActionSelection):
    def __init__(self):
        super().__init__(epsilon=0.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
