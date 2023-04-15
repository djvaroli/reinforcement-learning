from typing import Dict, Any

import torch
from torch import nn


class DQN(nn.Module):
    """Deep Q Network.
    """
    def __init__(
        self, 
        n_observations: int, 
        n_actions: int,
        hidden_size: int = 128,
        hidden_activation: nn.Module = nn.ReLU()
    ):
        """Initialize the DQN.

        Args:
            n_observations (int): number of observations (observed quantities) in the environment.
            n_actions (int): number of actions in the environment.
            hidden_size (int, optional): dimension of the hidden layers. Defaults to 128.
            hidden_activation (nn.Module, optional): activation function applied between hidden layers. Defaults to nn.ReLU().
        """
        super(DQN, self).__init__()
        self.hidden_activation = hidden_activation
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DQN.

        Args:
            x (torch.Tensor): tensor with shape (batch_size, n_observations).

        Returns:
            torch.Tensor: tensor with shape (batch_size, n_actions).
        """
        x = self.hidden_activation(self.layer1(x))
        x = self.hidden_activation(self.layer2(x))
        return self.layer3(x)
    
    def config(self) -> Dict[str, Any]:
        """Returns the configuration of the DQN.

        Returns:
            Dict[str, Any]: dictionary with the configuration of the DQN.
        """
        return {
            "hidden_size": self.hidden_size,
            "n_hidden_layers": 1,
            "hidden_activation": self.hidden_activation.__class__.__name__
        }