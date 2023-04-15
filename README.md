# RL Demos in PyTorch

## Cart Pole
Code based on [demo from PyTorch documentation.](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

### Training Agent
```python
from typing import List
import random
import numpy as np

import torch
from torch import optim
import matplotlib
import matplotlib.pyplot as plt
import math

from agent import CartPoleAgent
from schedules import EpsilonDecaySchedule

# set seed for reproducibility
SEED = 10000
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# configure necessary classes
eps_decay_schedule = EpsilonDecaySchedule(
    initial_value=0.9,
    final_value=0.05,
    decay_coefficient=1000
)

agent = CartPoleAgent(
    memory_size=10000,
    epsilon_schedule=eps_decay_schedule,
    tau=0.005
)

agent.init(
    optimizer=optim.AdamW(
        agent.policy_net_parameters(), 
        lr=1e-4, 
        amsgrad=True
    )
)

loss_history = agent.train(600, 128)
```