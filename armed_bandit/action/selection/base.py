from abc import ABC, abstractmethod

from jax import Array


class ActionSelection(ABC):

    @abstractmethod
    def select(self, random_key: Array, est_action_values: Array) -> Array: ...
