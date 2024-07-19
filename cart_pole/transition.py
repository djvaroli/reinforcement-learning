from dataclasses import dataclass
from typing import Callable, List, NamedTuple

import torch


class Transition(NamedTuple):
    """A transition is a tuple containing the
    state, action, reward and next state of an agent's interaction with the environment.
    """

    state: torch.FloatTensor
    action: torch.LongTensor
    reward: torch.FloatTensor
    next_state: torch.FloatTensor

    def as_tuple(self) -> tuple:
        """Returns the transition as a tuple.

        Returns:
            tuple: a tuple containing the state, action, reward and next state of the transition.
        """
        return self.state, self.action, self.reward, self.next_state

    def as_dict(self) -> dict:
        """Returns the transition as a dictionary.

        Returns:
            dict: a dictionary containing the state, action, reward and next state of the transition.
        """
        return self._asdict()


@dataclass
class TransitionBatch:
    transitions: List[Transition]

    def states_batch(
        self, filter_fn: Callable[[Transition], bool]
    ) -> torch.FloatTensor:
        """Returns a tensor batch of states from the transitions.

        Args:
            filter_fn (Callable[[Transition], bool]): a function that takes a transition and returns a boolean,
                indicating whether the transition should be included in the batch or not.

        Returns:
            torch.FloatTensor: a tensor batch of states, with shape (batch_size, state_size)
        """
        return torch.stack([t.state for t in self.transitions if filter_fn(t)])

    def actions_batch(
        self, filter_fn: Callable[[Transition], bool]
    ) -> torch.LongTensor:
        """Returns a tensor batch of actions from the transitions.

        Args:
            filter_fn (Callable[[Transition], bool]): a function that takes a transition and returns a boolean,
                indicating whether the transition should be included in the batch or not.

        Returns:
            torch.LongTensor: a tensor batch of actions, with shape (batch_size, action_size)
        """
        return torch.stack([t.action for t in self.transitions if filter_fn(t)])

    def rewards_batch(
        self, filter_fn: Callable[[Transition], bool]
    ) -> torch.FloatTensor:
        """Returns a tensor batch of rewards from the transitions.

        Args:
            filter_fn (Callable[[Transition], bool]): a function that takes a transition and returns a boolean,
                indicating whether the transition should be included in the batch or not.

        Returns:
            torch.FloatTensor: a tensor batch of rewards, with shape (batch_size, 1)
        """
        return torch.stack([t.reward for t in self.transitions if filter_fn(t)])

    def next_states_batch(
        self, filter_fn: Callable[[Transition], bool]
    ) -> torch.FloatTensor:
        """Returns a tensor batch of next states from the transitions.

        Args:
            filter_fn (Callable[[Transition], bool]): a function that takes a transition and returns a boolean,
                indicating whether the transition should be included in the batch or not.

        Returns:
            torch.FloatTensor: a tensor batch of next states, with shape (batch_size, state_size)
        """
        return torch.stack([t.next_state for t in self.transitions if filter_fn(t)])
