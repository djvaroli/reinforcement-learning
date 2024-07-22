from jax import Array
from jax import numpy as jnp

from .base import ActionSelection
from .utils import _argmax_break_ties


class UpperConfidenceBoundSelection(ActionSelection):
    def __init__(self, exploration_strength: float) -> None:
        super().__init__()
        self.exploration_strength = exploration_strength

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(c={self.exploration_strength})"

    def select(
        self, random_key: Array, est_action_values: Array, selection_count: Array
    ) -> Array:
        # offset by 1 to avoid taking the log of 0
        time_step_p1 = jnp.sum(selection_count) + 1

        # when selection_count is 0, the variance is infinite so we add a small offset
        # an action with a selection count of 0 is therefore likely to be selected just as we would expect
        eps_offset = 1e-5
        action_estimate_variance = jnp.sqrt(
            jnp.log(time_step_p1) / (selection_count + eps_offset)
        )

        return _argmax_break_ties(
            random_key,
            est_action_values + self.exploration_strength * action_estimate_variance,
        )
