from collections import deque
from copy import deepcopy, copy
from dataclasses import dataclass, InitVar
from functools import lru_cache, reduce
from itertools import chain
import numpy as np
import torch
from torch.nn.functional import mse_loss

from rlrd.memory import Memory
from rlrd.nn import PopArt, no_grad, copy_shared, exponential_moving_average, hd_conv
from rlrd.util import cached_property, partial
import rlrd.sac_models

@dataclass(eq=False)
class Agent:
    # Initializer variables
    Env: InitVar  # The environment the agent will interact with

    # Default attributes
    Model: type = rlrd.sac_models.Mlp  # SAC model class to be used
    OutputNorm: type = PopArt  # Normalization class to be used
    batchsize: int = 256  # Number of samples per training batch
    memory_size: int = 1000000  # Size of the replay buffer
    lr: float = 0.0003  # Learning rate for optimizers
    discount: float = 0.99  # Discount factor for future rewards
    target_update: float = 0.005  # Soft update parameter for target networks
    reward_scale: float = 5.  # Scaling factor for rewards
    entropy_scale: float = 1.  # Scaling factor for entropy bonus
    start_training: int = 10000  # Number of steps to collect before training starts
    device: str = None  # Device ('cuda' or 'cpu') to run the computation
    training_steps: float = 1.  # Training steps to perform per environment step

    # Cached property to get a version of the model that doesn't track gradients
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))
    
    # Total number of training updates performed
    # will be (len(self.memory)-start_training) * training_steps / training_interval
    total_updates: int = 0  
    # Total number of steps taken in the environment
    environment_steps: int = 0  

    def __post_init__(self, Env):
        # Initialize environment and model
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.Model(observation_space, action_space).to(self.device)
        self.model_target = no_grad(deepcopy(self.model))

        # Optimizers for actor and critics
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critics.parameters(), lr=self.lr)

        # Replay memory
        self.memory = Memory(self.memory_size, self.batchsize, self.device)

        # Output normalization for critic outputs
        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

    def act(self, state, obs, r, done, info, train=False):
        stats = []
        # Process the action based on the current state and observations
        state = self.model.reset() if state is None else state
        action, next_state, _ = self.model.act(state, obs, r, done, info, train=train)

        # Store transition in memory and possibly train the model
        if train:
            self.memory.append(np.float32(r), np.float32(done), info, obs, action)
            self.environment_steps += 1

            # Calculate the number of updates based on steps taken
            total_updates_target = (self.environment_steps - self.start_training) * self.training_steps
            stats = []
            while self.total_updates < int(total_updates_target):
                stats.append(self.train())
                self.total_updates += 1
        return action, next_state, stats

    def train(self):
        # Sample a batch of transitions from the memory
        obs, actions, rewards, next_obs, terminals = self.memory.sample()

        # Compute new actions from the current policy
        # outputs distribution object
        new_action_distribution = self.model.actor(obs)
        # samples using the reparametrization trick
        new_actions = new_action_distribution.rsample()

        # Compute critic loss on next state actions
        # outputs distribution object
        next_action_distribution = self.model_nograd.actor(next_obs)
        next_actions = next_action_distribution.sample()
        next_value = [c(next_obs, next_actions) for c in self.model_target.critics]
        # minimum action-value
        next_value = reduce(torch.min, next_value)
        # PopArt (not present in the original paper)
        next_value = self.outputnorm_target.unnormalize(next_value)
        # next_value = self.outputnorm.unnormalize(next_value)

        # Combine scaled rewards and entropy bonus
        # predict entropy rewards in a separate dimension from the normal rewards (not present in the original paper)
        next_action_entropy = - (1. - terminals) * self.discount * next_action_distribution.log_prob(next_actions)
        reward_components = torch.cat((
            self.reward_scale * rewards[:, None],
            self.entropy_scale * next_action_entropy[:, None],
        ), dim=1)  # shape = (batchsize, reward_components)

        value_target = reward_components + (1. - terminals[:, None]) * self.discount * next_value
        normalized_value_target = self.outputnorm.update(value_target)  # PopArt update and normalize

        # Compute losses for the critics
        values = [c(obs, actions) for c in self.model.critics]
        assert values[0].shape == normalized_value_target.shape and not normalized_value_target.requires_grad
        loss_critic = sum(mse_loss(v, normalized_value_target) for v in values)
        
        # update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Compute actor loss and update actor
        new_value = [c(obs, new_actions) for c in self.model.critics]
        new_value = reduce(torch.min, new_value)
        assert new_value.shape == (self.batchsize, 2)

        new_value = self.outputnorm.unnormalize(new_value)
        new_value[:, -1] -= self.entropy_scale * new_action_distribution.log_prob(new_actions)
        loss_actor = -self.outputnorm.normalize_sum(new_value.sum(1)).mean()
        
        # update actor
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Update target networks and normalizers using exponential moving average
        exponential_moving_average(self.model_target.critics.parameters(), self.model.critics.parameters(), self.target_update)
        exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.target_update)

        return {
            'loss_actor': loss_actor.detach(),
            'loss_critic': loss_critic.detach(),
            'outputnorm_reward_mean': self.outputnorm.mean[0],
            'outputnorm_entropy_mean': self.outputnorm.mean[-1],
            'outputnorm_reward_std': self.outputnorm.std[0],
            'outputnorm_entropy_std': self.outputnorm.std[-1],
            'memory_size': len(self.memory),
        }