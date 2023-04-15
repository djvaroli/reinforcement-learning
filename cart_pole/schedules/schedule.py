from abc import ABC, abstractmethod
from typing import Any


class Schedule(ABC):
    """Base class for all schedules.

    All schedules must implement the __call__ method, that takes at least an episode number as input,
    and returns a value.
    """

    @abstractmethod
    def __call__(self, episode: int, *args: Any, **kwds: Any) -> Any:
        return super().__call__(episode, *args, **kwds)
