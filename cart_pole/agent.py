from typing import Deque, List, Callable, Dict, Iterator, Optional
from collections import deque
import random
from itertools import count
from dataclasses import dataclass

import torch
from torch import nn
from torch import optim as optim
import numpy as np
import gymnasium as gym
import tqdm

from transition import Transition
from network import DQN


@dataclass
class CartPoleAgentConfig:

    env_name: str
    memory_size: int
    tau: float
    gamma: float
    dqn_hidden_size: int
    epsilon_policy: Callable[[int], float]
    gradient_ceiling: float


class ReplayMemory:
    def __init__(
        self, 
        capacity: int
    ) -> None:
        self.memory: Deque[Transition] = deque(list(), maxlen=capacity)
    
    def push(self, transition: Transition) -> None:
        """Push a transition to the replay memory.

        Args:
            transition (Transition): The transition to push.
        """
        self.memory.append(transition)
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions from the replay memory.

        Args:
            batch_size (int): number of transitions to sample.

        Raises:
            ValueError: If the batch size is greater than the replay memory size.

        Returns:
            List[Transition]: A list of sampled transitions.
        """
        if batch_size > len(self.memory):
            raise ValueError(f"Batch size ({batch_size}) is greater than the replay memory size ({len(self.memory)}).")
        
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)


class CartPoleAgent:
    def __init__(
        self,
        memory_size: int = 10000,
        dqn_hidden_size: int = 128,
        tau: float = 0.005,
        gamma: float = 0.99,
        epsilon_schedule: Callable[[int], float] = lambda steps_done: 0.5,
        gradient_ceiling: float = 100.0
    ) -> None:
        """Initialize the CartPole agent.

        Args:
            memory_size (int): The size of the replay memory. Defaults to 10000.
            dqn_hidden_size (int): The size of the hidden layers in the DQN. Defaults to 128.
            tau (float): The soft update parameter, which controls the rate at which the target network is updated. Defaults to 0.005.
            gamma (float): The discount factor, which controls the importance of future rewards. Defaults to 0.99.
            epsilon_schedule (Callable[[int], float]): a callable that takes the number of steps done and returns the epsilon value. 
                Defaults to lambda steps_done: 0.5.
            gradient_ceiling (float): Value to clip the gradients to. Defaults to 100.0.
        """
        self._env_name = "CartPole-v1"
        self._memory_size = memory_size
        self._steps_done = 0
        self._episode_durations = []
        self._loss_history: Dict[int, List[float]] = {}

        self.env = gym.make(self._env_name)

        # state has shape (4,) (x, x_dot, theta, theta_dot)
        state, info = self.env.reset()
        self._n_observations = len(state)

        # initialize the networks
        self.policy_net = DQN(self.n_observations, self.n_actions, dqn_hidden_size)
        self.target_net = DQN(self.n_observations, self.n_actions, dqn_hidden_size)

        # initialize the two networks to have the same weights as the policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.replay_memory = ReplayMemory(memory_size)

        self._epsilon_schedule = epsilon_schedule
        self.tau = tau
        self.gamma = gamma
        self.gradient_ceiling = gradient_ceiling
    
    @property
    def config(self) -> CartPoleAgentConfig:
        return CartPoleAgentConfig(
            env_name=self.env_name,
            memory_size=self._memory_size,
            dqn_hidden_size=self.policy_net.hidden_size,
            tau=self.tau,
            gamma=self.gamma,
            epsilon_policy=self._epsilon_schedule,
            gradient_ceiling=self.gradient_ceiling
        )
    
    @property
    def n_actions(self) -> int:
        return self.env.action_space.n

    @property
    def n_observations(self) -> int:
        return self._n_observations

    def get_epsilon_threshold(self, steps_done: int) -> float:
        """Returns a value between 0 and 1, to be used as the epsilon threshold
        for the epsilon-greedy policy.

        Args:
            steps_done (int): The number of steps taken so far.

        Returns:
            float: The epsilon threshold.
        """
        return self._epsilon_schedule(steps_done)
    
    def select_action(self, state: torch.Tensor, epsilon: float) -> torch.LongTensor:
        """Select an action using the policy network or a random action depending
        on the epsilon threshold.

        Args:
            state (torch.Tensor): The current state of the environment.
            epsilon (float): Value of threshold for epsilon-greedy policy. 
                If a random number is greater than epsilon, the policy network is used to select the action.
                Otherwise, a random action is selected from the replay memory.
        
        Returns:
            torch.LongTensor: The action to take.
        """
        if random.random() > epsilon:
            # use the policy network to select the action
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # select a random action
            action_index: int = self.env.action_space.sample()
            return torch.tensor([[action_index]], dtype=torch.long)

    def policy_net_parameters(self) -> Iterator[nn.Parameter]:
        """Returns the parameters of the policy network.
        """
        return self.policy_net.parameters()
    
    def init(
        self, 
        optimizer: optim.Optimizer, 
        loss_fn: nn.Module = nn.SmoothL1Loss()
    ) -> "CartPoleAgent":
        """Initialize the agent, by setting the optimizer and loss function.

        Args:
            optimizer (optim.Optimizer): The optimizer to use.
            loss_fn (nn.Module, optional): The loss function to use. Defaults to nn.SmoothL1Loss().
        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        return self
    
    def _update_target_net_weights(self, tau: float) -> None:
        """Update the target network weights using a soft update approach.

        Parameters:
            tau (float): The soft update coefficient. When ``tau`` is 1, the target network weights
                will be updated to be equal to the policy network weights. When ``tau`` is 0, the target
                network weights will not be updated.
        """
        if tau > 1.0 or tau < 0.0:
            raise ValueError("``tau`` must be in the range [0, 1].")
        
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        # target net weights are updated using a soft update approach
        # this helps to stabilize the training process
        for key in target_net_state_dict:
            target_net_state_dict[key] = tau * policy_net_state_dict[key] + (1 - tau) * target_net_state_dict[key]

        self.target_net.load_state_dict(target_net_state_dict)

    def _increment_steps_done(self) -> None:
        """Increment the number of steps taken so far.
        """
        self._steps_done += 1
    
    def optimize_model(self, batch_size: int = 128) -> float:
        """Samples a batch of transitions from the replay memory and uses it to train the policy network.

        Args:
            batch_size (int, optional): The size of the batch to sample from the replay memory. Defaults to 128.

        Returns:
            float: The computed value of the Huber loss.
        """
        if len(self.replay_memory) < batch_size:
            return np.inf

        sampled_transitions = self.replay_memory.sample(batch_size)

        # this takes a list of individual transitions and converts it to a single transition
        # where `state` is a tuple of the `state` from each individual transition
        # `action` is a tuple of the `action` from each individual transition and so on
        # FIXME: this works but the typing is wrong
        training_batch = Transition(*zip(*sampled_transitions))

        # for final states V(s) = max_a Q(s, a) = 0 by definition
        non_final_mask = torch.tensor(
            tuple(
                map(lambda s: s is not None, training_batch.next_state)
            ), 
            dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s for s in training_batch.next_state if s is not None]
        )
        
        # sate_batch is a tensor of shape (batch_size, n_observations)
        state_batch = torch.cat(training_batch.state)

        # action_batch is a tensor of shape (batch_size, 1)
        action_batch = torch.cat(training_batch.action)

        # reward_batch is a tensor of shape (batch_size, 1)
        reward_batch = torch.cat(training_batch.reward)

        # compute the Q(s, a) values for the current state
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size)
        # computer the max_a(Q(s', a)) values for the next state
        # for final states V(s) = max_a Q(s, a) = 0 by definition
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # recall that Q(s, a) = r + gamma * max_a' Q(s', a'
        expected_state_action_values = reward_batch + (next_state_values * self.gamma)
        loss: torch.Tensor = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        # minimize Huber loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # clip gradients to prevent them from exploding
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.gradient_ceiling)
        self.optimizer.step()

        return loss.item()
    
    def train_episode(self,  batch_size: int, episode: Optional[int] = None) -> List[float]:
        """Trains the agent for a single episode.

        Args:
            batch_size (int): The size of the batch to sample from the replay memory.
            episode (Optional[int], optional): The episode number. Defaults to None.
        
        Returns:
            List[float]: A list of the loss values computed during the episode.
        """
        episode_loss_history = list()
        state, info = self.env.reset()

        # convert the state to a tensor
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        for t in count():
            # select action
            epsilon = self.get_epsilon_threshold(self._steps_done)
            action: torch.LongTensor = self.select_action(state, epsilon)
            self._increment_steps_done()

            # take action and observe the next state and reward
            observation, reward, terminated, truncated, _ = self.env.step(action.item())
            
            # convert reward to a tensor
            reward = torch.tensor([reward], dtype=torch.float32)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            self.replay_memory.push(Transition(state, action, reward, next_state))
            state = next_state

            step_loss = self.optimize_model(batch_size)
            self._update_target_net_weights(tau=self.tau)

            episode_loss_history.append(step_loss)

            if done:
                self._episode_durations.append(t + 1)
                break
        
        return episode_loss_history

    def train(self, n_episodes: int, batch_size: int):
        """Trains the agent for a given number of episodes.

        Args:
            n_episodes (int): Number of episodes to train for.
            batch_size (int): The size of the batch to sample from the replay memory.
        """

        with tqdm.trange(n_episodes, desc="Training", unit="episode") as progress_bar:
            for episode in range(n_episodes):
                self._loss_history[episode] = self.train_episode(batch_size, episode)
                mean_episode_loss = np.mean(self._loss_history[episode])
                progress_bar.set_postfix({f"{self.loss_fn.__class__.__name__}": round(mean_episode_loss, 4)})
                progress_bar.update(1)

