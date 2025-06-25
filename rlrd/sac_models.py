from dataclasses import InitVar, dataclass

import gym
import torch
import numpy as np
from torch.nn.functional import leaky_relu, relu
from gym.spaces import Box, Discrete
import torch.nn as nn

from rlrd.util import collate, partition
from torch.nn import Linear, Sequential, ReLU, ModuleList, Module
from rlrd.nn import TanhNormalLayer, SacLinear, big_conv

class ActorModule(Module):
    device = 'cpu'
    actor: callable

    # noinspection PyMethodOverriding
    def to(self, device):
        """keeps track which device this module has been moved to"""
        self.device = device
        return super().to(device=device)

    def reset(self):
        """Initialize the hidden state. This will be collated before being fed to the actual model and thus should be a structure of numpy arrays rather than torch tensors."""
        return np.array(())  # just so we don't get any errors when collating and partitioning

    def act(self, state, obs, r, done, info, train=False):
        """allows this module to be used with gym.Env
        converts inputs to torch tensors and converts outputs to numpy arrays"""
        obs = collate([obs], device=self.device)
        with torch.no_grad():
            # returns actions distribution with tanh applied
            action_distribution = self.actor(obs)
            # Action normalizer in the env
            action = action_distribution.sample() if train else action_distribution.sample_deterministic()
            
            # HINT: Added by me
            # action should be 2D to apply the log_prob
            """
            action = action.squeeze(0)
            action = torch.tensor(self.min_action).to(self.device) +\
                (action + 1) * (torch.tensor(self.max_action).to(self.device) -\
                 torch.tensor(self.min_action).to(self.device)) / 2 
                 
            log_probs = action_distribution.log_prob(action)
            def safe_log(x):
                # Safe logarithm implementation
                eps = 1e-6
                return torch.log(torch.clamp(x, min=eps))
            
            log_probs -= safe_log(1 - action.pow(2) + self.reparam_noise)
            # log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
            # FIXME: check if it is correct to sum all actions log_probs
            log_probs = log_probs.sum(-1, keepdim=True)"""
            
        action, = partition(action)
        return action, state, []


class MlpActionValue(Sequential):
    def __init__(self, dim_obs, dim_action, hidden_units):
        super().__init__(
            SacLinear(dim_obs + dim_action, hidden_units), ReLU(),
            SacLinear(hidden_units, hidden_units), ReLU(),
            Linear(hidden_units, 2)
        )

    # noinspection PyMethodOverriding
    def forward(self, obs, action):
        x = torch.cat((*obs, action), -1)
        return super().forward(x)


class MlpPolicy(Sequential):
    def __init__(self, dim_obs, dim_action, hidden_units):
        self.dim_obs = dim_obs
        super().__init__(
            SacLinear(dim_obs, hidden_units), ReLU(),
            SacLinear(hidden_units, hidden_units), ReLU(),
            TanhNormalLayer(hidden_units, dim_action)
        )

    # noinspection PyMethodOverriding
    def forward(self, obs):
        # return super().forward(torch.cat(obs, -1))  # XXX
        return super().forward(torch.cat((obs,), -1)).view(-1, )  # Pass obs as a tuple


class Mlp(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256, num_critics: int = 2):
        super().__init__()
        # assert isinstance(observation_space, gym.spaces.Tuple)
        # dim_obs = sum(space.shape[0] for space in observation_space)
        
        # Function to calculate the total dimensions of the observation space
        def calculate_dim_obs(observation_space):
            dim_obs = 0
            for space in observation_space:
                if isinstance(space, Box):
                    dim_obs += space.shape[0]
                elif isinstance(space, Discrete):
                    dim_obs += 1  # Discrete space adds a single dimension
                else:
                    raise TypeError(f"Unsupported space type: {type(space)}")
            return dim_obs
        
        # dim_obs = calculate_dim_obs(observation_space)
        
        dim_obs = observation_space.shape[-1]
        dim_action = action_space.shape[0]
        self.min_action = action_space.low
        self.max_action= action_space.high
        self.critics = ModuleList(MlpActionValue(dim_obs, dim_action, hidden_units) for _ in range(num_critics))
        self.actor = MlpPolicy(dim_obs, dim_action, hidden_units)
        self.critic_output_layers = [c[-1] for c in self.critics]
        

# === recurrent neural network models =======================================================================================================
class LSTMActorModule(Module):
    device = 'cpu'
    actor: callable

    # noinspection PyMethodOverriding
    def to(self, device):
        """keeps track which device this module has been moved to"""
        self.device = device
        return super().to(device=device)

    def reset(self):
        """Initialize the hidden state. This will be collated before being fed to the actual model and thus should be a structure of numpy arrays rather than torch tensors."""
        return np.array(())  # just so we don't get any errors when collating and partitioning

    def act(self, state, obs, r, done, info, train=False):
        """allows this module to be used with gym.Env
        converts inputs to torch tensors and converts outputs to numpy arrays"""
        obs = collate([obs], device=self.device)
        with torch.no_grad():
            # returns actions distribution with tanh applied
            action_distribution = self.actor(obs)
            # Action normalizer in the env
            action = action_distribution.sample() if train else action_distribution.sample_deterministic()
            
        action, = partition(action)
        return action, state, []


class LSTMActionValue(nn.Sequential):
    def __init__(self, dim_obs, dim_action, hidden_units, lstm_hidden_units):
        super().__init__(
            nn.LSTM(input_size=dim_obs + dim_action, hidden_size=lstm_hidden_units, batch_first=True),
            nn.Linear(lstm_hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, 2)
        )

    def forward(self, obs, action):
        x = torch.cat((obs, action), -1)
        x, _ = self[0](x.unsqueeze(1))  # LSTM expects 3D input (batch, seq, feature)
        x = x.contiguous().view(x.size(0), -1)
        x = self[1](x)
        x = relu(x)
        x = self[2](x)
        x = relu(x)
        x = self[3](x)
        return x


class LSTMPolicy(nn.Sequential):
    def __init__(self, dim_obs, dim_action, hidden_units, lstm_hidden_units):
        super().__init__(
            nn.LSTM(input_size=dim_obs, hidden_size=lstm_hidden_units, batch_first=True),
            nn.Linear(lstm_hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            TanhNormalLayer(hidden_units, dim_action)
        )

    def forward(self, obs):
        x, _ = self[0](obs.unsqueeze(1))  # LSTM expects 3D input (batch, seq, feature)
        x = x.contiguous().view(x.size(0), -1)
        x = self[1](x)
        x = relu(x)
        x = self[2](x)
        x = relu(x)
        x = self[3](x)
        return x


class LSTM(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256, lstm_hidden_units: int = 256, num_critics: int = 2):
        super().__init__()
        
        dim_obs = observation_space.shape[-1]
        dim_action = action_space.shape[0]
        self.min_action = action_space.low
        self.max_action = action_space.high
        self.critics = ModuleList(LSTMActionValue(dim_obs, dim_action, hidden_units, lstm_hidden_units) for _ in range(num_critics))
        self.actor = LSTMPolicy(dim_obs, dim_action, hidden_units, lstm_hidden_units)
        self.critic_output_layers = [c[-1] for c in self.critics]

        

# === convolutional models =======================================================================================================
class ConvActor(Module):
    def __init__(self, observation_space, action_space, hidden_units: int = 512, Conv: type = big_conv):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        (img_sp, vec_sp), *aux = observation_space

        self.conv = Conv(img_sp.shape[0])
        with torch.no_grad():
            conv_size = self.conv(torch.zeros((1, *img_sp.shape))).view(1, -1).size(1)
            print("conv_size =", conv_size)
        self.lin1 = torch.nn.Linear(
            conv_size + vec_sp.shape[0] + sum(sp.shape[0] for sp in aux),
            hidden_units)
        self.lin2 = torch.nn.Linear(
            hidden_units + vec_sp.shape[0] + sum(sp.shape[0] for sp in aux),
            hidden_units
        )
        self.output_layer = TanhNormalLayer(hidden_units, action_space.shape[0])

    def forward(self, observation):
        (x, vec), *aux = observation
        x = x.type(torch.float32)
        x = x / 255 - 0.5
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = leaky_relu(self.lin1(torch.cat((x, vec, *aux), -1)))
        x = leaky_relu(self.lin2(torch.cat((x, vec, *aux), -1)))
        x = self.output_layer(x)
        return x


class ConvCritic(Module):
    def __init__(self, observation_space, action_space, hidden_units: int = 512, Conv: type = big_conv):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        (img_sp, vec_sp), *aux = observation_space

        self.conv = Conv(img_sp.shape[0])
        with torch.no_grad():
            conv_size = self.conv(torch.zeros((1, *img_sp.shape))).view(1, -1).size(1)

        self.lin1 = torch.nn.Linear(
            conv_size + vec_sp.shape[0] + sum(sp.shape[0] for sp in aux) + action_space.shape[0],
            hidden_units)
        self.lin2 = torch.nn.Linear(
            hidden_units + vec_sp.shape[0] + sum(sp.shape[0] for sp in aux) + action_space.shape[0],
            hidden_units
        )
        self.output_layer = torch.nn.Linear(hidden_units, 2)
        self.critic_output_layers = self.output_layer,

    def forward(self, observation, a):
        (x, vec), *aux = observation
        x = x.type(torch.float32)
        x = x / 255 - 0.5
        x = self.conv(x)
        print(x.shape)
        # x = x.view(x.size(0), -1)
        x = leaky_relu(self.lin1(torch.cat((x, vec, *aux, a), -1)))
        x = leaky_relu(self.lin2(torch.cat((x, vec, *aux, a), -1)))
        x = self.output_layer(x)
        return x


class ConvModel(ActorModule):
    def __init__(self, observation_space, action_space, num_critics: int = 2, hidden_units: int = 256, Conv: type = big_conv):
        super().__init__()
        self.actor = ConvActor(observation_space, action_space, hidden_units, Conv)
        self.critics = ModuleList(ConvCritic(observation_space, action_space, hidden_units, Conv) for _ in range(num_critics))
        self.critic_output_layers = sum((c.critic_output_layers for c in self.critics), ())
