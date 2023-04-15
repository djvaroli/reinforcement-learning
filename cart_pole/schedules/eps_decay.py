import math


class EpsilonDecaySchedule:
    """Epsilon decay schedule.
    Computes the value of epsilon at a given episode, according to the following formula:
    epsilon = final_value + (initial_value - final_value) * exp(-decay_coefficient * episode)
    """

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        decay_coefficient: float,
    ) -> None:
        """Initializes the schedule.

        Args:
            initial_value (float): the initial value of epsilon, or the value returned when episode = 0.
            final_value (float): the final value of epsilon, or the value returned when episode -> infinity.
            decay_coefficient (float): the decay coefficient of epsilon. The higher the coefficient, the faster
                the value of epsilon will decay.
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_coefficient = decay_coefficient

    def __call__(self, episode: int) -> float:
        range_ = self.initial_value - self.final_value
        return self.final_value + range_ * math.exp(
            -1.0 * self.decay_coefficient * episode
        )
