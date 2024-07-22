from jax import Array
from jax import numpy as jnp
from jax import random as jrnd


def _argmax_break_ties(random_key: Array, array: Array) -> Array:
    """Takes the argmax of an array, breaking ties randomly.

    Args:
        random_key (Array): random key for reproducibility
        array (Array): array to take the argmax of.

    Returns:
        Array: index of the maximum value in the array,
            or one of the indices in the case of ties.
    """
    val_max = jnp.max(array)
    ties = jnp.where(array == val_max)[0]
    return jrnd.choice(random_key, ties)
