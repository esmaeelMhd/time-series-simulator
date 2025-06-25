import os
import sys
import warnings
import pickle
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, InitVar
from typing import Any
import gym

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nn import no_grad, exponential_moving_average
from dataclasses import dataclass, field, fields

from buffer import ReplayBuffer
from buffer import TrajMemoryNoHidden
from networks import ActorNetwork, CriticNetwork, ValueNetwork

from models.LSTM import LSTMModel
from envs.utils import Args

from dcac_models import Mlp
from functools import reduce
from torch.nn.functional import mse_loss

from rlrd.memory import Memory
from rlrd.nn import PopArt, no_grad, copy_shared, exponential_moving_average, hd_conv
from rlrd.util import cached_property, partial
import rlrd.sac_models

@dataclass
class SACAgent:
    """
    A class that encapsulates the training agent which can handle both LSTM-based and standard neural network models
    for reinforcement learning tasks. This class is responsible for managing network architectures, memory buffers,
    and optimization setups.
    
    Attributes:
        lr_actor (float): Learning rate for the actor network.
        lr_critic (float): Learning rate for the critic network.
        input_dims (tuple): Dimensions of inputs to the network.
        tau (float): Soft update parameter for the target networks.
        env (any): environment object which must have an action_space attribute.
        env_id (str): Identifier for the environment, used in naming models.
        gamma (float): Discount factor for future rewards, default=0.99.
        n_actions (int): Number of actions in the action space, default=2.
        max_mem_size (int): Maximum size of the replay buffer, default=1000000.
        layer1_size (int): Number of neurons in the first hidden layer, default=256.
        layer2_size (int): Number of neurons in the second hidden layer, default=256.
        batch_size (int): Number of samples per batch during training, default=100.
        reward_reward_scale (int): Scaling factor for rewards, default=2.
        init_policy (str): Initial policy filename, default an empty string.
        actor_net (str): Type of actor network, default an empty string.
        device (str): Device to run the computations ('cuda' or 'cpu'), default='cuda'.
        agent_args (dict): Additional arguments for agents, default an empty dict.
    
        memory (any): Replay buffer initialized post instantiation.
        actor (any): The actor network initialized based on the actor_net attribute.
        critic_1 (any): The first critic network.
        critic_2 (any): The second critic network.
        value (any): The value network.
        target_value (any): The target value network.
        reward_scale (float): Reward scaling, set post instantiation.
    
    Methods:
        __post_init__(): Handles post-initialization tasks including network setup.
        setup_lstm_actor_critic(): Configures networks and optimization for LSTM-based models.
        setup_standard_actor_critic(): Sets up standard neural networks and optimizers.
        setup_critic(name: str, LSTM: bool=False): Helper to initialize critic networks.
        setup_value_network(target: bool=False, LSTM: bool=False): Helper to initialize value networks.
    """
    env: Any
    input_dims: tuple
    n_actions: int
    env_id: str = 'time_series_env'
    lr_actor: float = 0.0003
    lr_critic: float = 0.0003
    gamma: float = 0.99
    max_mem_size: int = 1000000
    layer1_size: int = 256
    layer2_size: int = 256
    batch_size: int = 100
    reward_reward_scale: int = 2
    init_policy: str = ''
    actor_net: str = ''
    reward_scale: float = 2
    tau: float = 1.0
    const_delay: int = 4
    use_gpu: bool = True
    device: str = 'cuda'
    agent_args: dict = field(default_factory=dict)
    memory: any = field(init=False)
    actor: any = field(init=False)
    critic_1: any = field(init=False)
    critic_2: any = field(init=False)
    value: any = field(init=False)
    target_value: any = field(init=False)

    def __post_init__(self):
        """Handle complex initializations that are not suitable for default dataclass constructor."""
        # Setup the device based on GPU usage
        self.device = T.device('cuda') if T.cuda.is_available() and self.use_gpu else T.device('cpu')
        if self.device == 'cuda':
            self.device_ids = ','.join(str(i) for i in range(T.cuda.device_count()))
            self.use_multi_gpu = T.cuda.device_count() > 1
        else:
            self.device_ids = ''
            self.use_multi_gpu = False 
            
        if self.input_dims is None:
            self.input_dims = self.env.observation_space.shape
        if self.n_actions is None:
            self.n_actions = self.env.action_space.shape[-1]
            
        self.memory = ReplayBuffer(self.max_mem_size, self.input_dims, self.n_actions)
        if self.actor_net.lower() == 'lstm':
            self.setup_lstm_actor_critic()
        else:
            self.setup_standard_actor_critic()
            
        self.update_network_parameters(tau=1)
    
    @staticmethod
    def from_namespace(args, **kwargs):
        """Create an instance from a namespace, allowing for manual attribute overrides or additions."""
        valid_keys = set(f.name for f in fields(SACAgent))
        filtered_args = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        
        # Override filtered_args with manually set attributes from kwargs
        filtered_args.update(kwargs)
        
        return SACAgent(**filtered_args)

    def setup_lstm_actor_critic(self):
        """Setup the LSTM based actor and critic networks along with related components."""
        ARGS_PATH = './policy_args/'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with open(ARGS_PATH + self.init_policy + '/args.pkl', 'rb') as file:
                model_args = pickle.load(file)
        model = LSTMModel(model_args).float()
        try:
            model.load_state_dict(
                T.load(os.path.join('./policy_checkpoints/' + self.init_policy, 'checkpoint.pth'), map_location=self.device))
        except OSError as e:
            print(f"Unable to open {self.checkpoint_file}: {e}", file=sys.stderr)
            return
        
        model.eval()
        if model.lstm.input_size != self.input_dims[-1]:
            print('The input sizes of the loaded model and environment are not the same, initializing new model...')
            args = Args(use_gpu=True, in_features=self.input_dims[-1], hidden_dim=256, layer_dim=1, 
                        batch_first=True, dropout=0, seq_len=self.const_delay, out_features=self.n_actions, pred_len=1)
            model = LSTMModel(args).float()
            
        self.actor = self.setup_actor('actor_lstm', LSTM=True, model=model)
        self.critic_1 = self.setup_critic('critic_1_lstm', LSTM=True)
        self.critic_2 = self.setup_critic('critic_2_lstm', LSTM=True)
        self.value = self.setup_value_network('value_lstm', LSTM=True)
        self.target_value = self.setup_value_network('target_value_lstm', LSTM=True)

    def setup_standard_actor_critic(self):
        """Initialize standard feed-forward neural networks for the actor and critics."""        
        self.actor = self.setup_actor('actor')
        self.critic_1 = self.setup_critic('critic_1')
        self.critic_2 = self.setup_critic('critic_2')
        self.value = self.setup_value_network('value')
        self.target_value = self.setup_value_network('target_value')
    
    def setup_actor(self, name, LSTM=False, model=None):
        """Setup the actor network."""
        actor = ActorNetwork(lr_actor=self.lr_actor, 
                             input_dims=self.input_dims, 
                             fc1_dims=self.layer1_size,
                             fc2_dims=self.layer2_size, 
                             min_action=self.env.action_space.low, 
                             max_action=self.env.action_space.high,
                             n_actions=self.n_actions, 
                             name=self.env_id+name, 
                             init_policy=self.init_policy,
                             use_lstm=LSTM,
                             lstm_output_dims=self.n_actions, 
                             model=model)
        
        return actor
    
    def setup_critic(self, name, LSTM=False):
        """Setup the critic network."""
        critic = CriticNetwork(lr_critic=self.lr_critic, 
                               input_dims=self.input_dims, 
                               fc1_dims=self.layer1_size,
                               fc2_dims=self.layer2_size, 
                               n_actions=self.n_actions,
                               name=self.env_id+name,
                               init_policy=self.init_policy,
                               use_lstm=LSTM,
                               lstm_hidden_dim=256)
        
        return critic
    
    def setup_value_network(self, name, LSTM=False):
        """Setup the value network."""
        value = ValueNetwork(lr_critic=self.lr_critic, 
                             input_dims=self.input_dims, 
                             fc1_dims=self.layer1_size, 
                             fc2_dims=self.layer2_size, 
                             name=self.env_id+name,
                             init_policy='', 
                             use_lstm=LSTM,
                             lstm_hidden_dim=256)
        
        return value        

    def choose_action(self, observation):
        state = T.Tensor(np.array([observation])).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        # actions, _ = self.actor.sample_mvnormal(state)
        # actions is an array of arrays due to the added dimension in state
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)

        # Value loss
        value = self.value(state).view(-1)
        value_ = self.target_value(new_state).view(-1)
        value_[done] = 0.0

        new_actions, log_probs = self.actor.sample_normal(
            state, reparameterize=False)
        # new_actions, log_probs = self.actor.sample_mvnormal(state, reparameterize=False)
        # log_probs = [batch_size, 1, n_actions]
        log_probs = log_probs.view(-1)
        
        q1_new_policy = self.critic_1.forward(state, new_actions)
        q2_new_policy = self.critic_2.forward(state, new_actions)

        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * (F.mse_loss(value, value_target))
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Actor loss
        new_actions, log_probs = self.actor.sample_normal(
            state, reparameterize=True)
        # new_actions, log_probs = self.actor.sample_mvnormal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, new_actions)
        q2_new_policy = self.critic_2.forward(state, new_actions)

        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)

        if self.actor_net == 'LSTM':
            self.actor.train()
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            # Gradient clipping to prevent exploding gradient problem
            T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor.optimizer.step()
        else:
            self.actor.train()
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.reward_scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_new_statedict = dict(target_value_params)
        value_new_statedict = dict(value_params)

        for name in value_new_statedict:
            value_new_statedict[name] = tau*value_new_statedict[name].clone() + \
                (1-tau)*target_value_new_statedict[name].clone()

        self.target_value.load_state_dict(value_new_statedict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

@dataclass(eq=0)
class SACAgentRD:
    # Initializer variables
    env: InitVar  # The environment the agent will interact with   
    agent_args: dict = field(default_factory=dict)
    env_id: str = 'time_series_env'
    init_policy: str = ''
    actor_net: str = ''
    const_delay: int = 4
    use_gpu: bool = True
    layer_size: int = 256

    # Default attributes
    Model: type = rlrd.sac_models.Mlp  # SAC model class to be used
    OutputNorm: type = PopArt  # Normalization class to be used
    batch_size: int = 256  # Number of samples per training batch
    memory_size: int = 1000000  # Size of the replay buffer
    lr: float = 0.0003  # Learning rate for optimizers
    discount: float = 0.99  # Discount factor for future rewards (gamma)
    target_update: float = 0.005  # Soft update parameter for target networks (tau)
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

    def __post_init__(self, env):
        # Initialize environment and model
        observation_space, action_space = env.observation_space, env.action_space
        self.device = self.device or ("cuda" if T.cuda.is_available() else "cpu")
        self.model = self.Model(observation_space, action_space).to(self.device)
        self.model_target = no_grad(deepcopy(self.model))

        # Optimizers for actor and critics
        self.actor_optimizer = T.optim.Adam(self.model.actor.parameters(), lr=self.lr)
        self.critic_optimizer = T.optim.Adam(self.model.critics.parameters(), lr=self.lr)

        # Replay memory
        self.memory = Memory(self.memory_size, self.batch_size, self.device)

        # Output normalization for critic outputs
        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

    @staticmethod
    def from_namespace(args, **kwargs):
        """Create an instance from a namespace, allowing for manual attribute overrides or additions."""
        valid_keys = set(f.name for f in fields(SACAgentRD))
        filtered_args = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        
        # Override filtered_args with manually set attributes from kwargs
        filtered_args.update(kwargs)
                
        return SACAgentRD(**filtered_args)
        
    def act(self, state, obs, r, done, trunc, info, train=False):
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
        print('############ Training step ...')
        # Sample a batch of transitions from the memory
        obs, actions, rewards, next_obs, terminals = self.memory.sample()
        
        z_obs = np.array(obs[0].detach().cpu())
        z_next_obs = np.array(next_obs[0].detach().cpu())
        z_actions = np.array(actions.detach().cpu())
        z_rewards = np.array(rewards.detach().cpu())
        z_terminals = np.array(terminals.detach().cpu())

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
        next_value = reduce(T.min, next_value)
        # PopArt (not present in the original paper)
        next_value = self.outputnorm_target.unnormalize(next_value)
        # next_value = self.outputnorm.unnormalize(next_value)

        # Combine scaled rewards and entropy bonus
        # predict entropy rewards in a separate dimension from the normal rewards (not present in the original paper)
        next_action_entropy = - (1. - terminals) * self.discount * next_action_distribution.log_prob(next_actions)
        reward_components = T.cat((
            self.reward_scale * rewards[:, None],
            self.entropy_scale * next_action_entropy[:, None],
        ), dim=1)  # shape = (batch_size, reward_components)

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

        # Compute actor loss and update actor (compute values based on the updataed critics for new actions)
        new_value = [c(obs, new_actions) for c in self.model.critics]
        new_value = reduce(T.min, new_value)
        assert new_value.shape == (self.batch_size, 2)

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
    
class DCACAgent():
    def __init__(self, batch_size=256, memory_size=1000000, lr=0.0003, discount=0.99, target_update=0.005, 
                 reward_reward_scale=5., entropy_reward_scale=1., start_training=10000, device=None, training_steps=1):
        super(DCACAgent, self).__init__()
        Model: type = Mlp
        loss_alpha: float = 0.2
        rtac: bool = False

    def __post_init__(self, env):
        with env() as env:
            observation_space, action_space = env.observation_space, env.action_space
            self.sup_obs_delay = env.obs_delay_range.stop
            self.sup_act_delay = env.act_delay_range.stop
            self.act_buf_size = self.sup_obs_delay + self.sup_act_delay - 1
            self.old_act_buf_size = deepcopy(self.act_buf_size)
            if self.rtac:
                self.act_buf_size = 1

        assert self.device is not None
        # or ("cuda" if T.cuda.is_available() else "cpu")
        device = self.device
        model = self.Model(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(
            self.model_target.critic_output_layers)

        self.optimizer = T.optim.Adam(self.model.parameters(), lr=self.lr)
        # TrajMemoryNoHidden: we store the obs returned by the wrapped env for the length of history which is act_buf_size
        self.memory = TrajMemoryNoHidden(
            self.memory_size, self.batch_size, device, history=self.act_buf_size)

        # lists that contains self.act_buf_size number of None elements
        self.traj_new_actions = [None, ] * self.act_buf_size
        self.traj_new_actions_detach = [None, ] * self.act_buf_size
        self.traj_new_actions_log_prob = [None, ] * self.act_buf_size
        self.traj_new_actions_log_prob_detach = [None, ] * self.act_buf_size
        self.traj_new_augm_obs = [None, ] * (self.act_buf_size + 1)

        self.is_training = False

    def train(self):
        # sample a trajectory of length self.act_buf_size
        # NB: when terminals is True, the terminal augmented state is the last one of the trajectory (this is ensured by the sampling procedure)

        # TODO: act_traj is useless, it could be removed from the replay memory
        # FIXME: the profiler indicates that memory is inefficient, optimize

        # It also samples a batch from Memory
        augm_obs_traj, act_traj, rew_traj, terminals = self.memory.sample()

        batch_size = terminals.shape[0]

        # value of the first augmented state:
        values = [c(augm_obs_traj[0]).squeeze() for c in self.model.critics]

        # nstep_len is the number of valid transitions of the sampled sub-trajectory, not counting the first one which is always valid since we consider the action delay to be always >= 1.
        # nstep_len will be e.g. 0 in the rtrl setting (an action delay of 0 here means an action delay of 1 in the paper).

        # int_tens_type: obs delay type (int)
        int_tens_type = obs_del = augm_obs_traj[0][2].dtype
        ones_tens = T.ones(batch_size, device=self.device,
                           dtype=int_tens_type, requires_grad=False)

        if not self.rtac:
            nstep_len = ones_tens * (self.act_buf_size - 1)
            # we don't care about the delay of the first observation in the trajectory, but we care about the last one
            for i in reversed(range(self.act_buf_size)):
                obs_del = augm_obs_traj[i + 1][2]  # observation delay (alpha)
                act_del = augm_obs_traj[i + 1][4]  # action_delay (beta)
                tot_del = obs_del + act_del
                # TODO: the last iteration is useless
                # If an element in tot_del is less than or equal to i, nstep_len
                # at that position is updated to be (i - 1). Otherwise, nstep_len remains urnchanged
                nstep_len = T.where(
                    (tot_del <= i), ones_tens * (i - 1), nstep_len)
            nstep_max_len = T.max(nstep_len)
            nstep_min_len = T.min(nstep_len)
            assert nstep_min_len >= 0, "Each total delay must be at least 1 (instantaneous turn-based RL not supported)"
            # each row contains zeros except for the position indicated by the corresponding element in nstep_len, which is set to 1.0
            # scatter_: (in-place) fills the tensor created by T.zeros with values specified by the last argument (1.) at the indices provided by nstep_len.unsqueeze(1).long().
            nstep_one_hot = T.zeros(len(nstep_len), nstep_max_len + 1, device=self.device,
                                    requires_grad=False).scatter_(1, nstep_len.unsqueeze(1).long(), 1.)
        # RTAC is equivalent to doing only 1-step backups (i.e. nstep_len==0)
        else:
            nstep_len = T.zeros(batch_size, device=self.device,
                                dtype=int_tens_type, requires_grad=False)
            nstep_max_len = T.max(nstep_len)
            nstep_one_hot = T.zeros(len(nstep_len), nstep_max_len + 1, device=self.device,
                                    requires_grad=False).scatter_(1, nstep_len.unsqueeze(1).long(), 1.)
            # the way the replay memory works, RTAC will never encounter terminal states for buffers of more than 1 action
            terminals = terminals if self.act_buf_size == 1 else terminals * 0.0

        # use the current policy to compute a new trajectory of actions of length self.act_buf_size
        for i in range(self.act_buf_size + 1):
            # compute a new action and update the corresponding *next* augmented observation:
            augm_obs = augm_obs_traj[i]
            if i > 0:
                act_slice = tuple(
                    self.traj_new_actions[self.act_buf_size - i:self.act_buf_size])
                # obs + action slice and action buffer + delays
                augm_obs = augm_obs[:1] + \
                    ((act_slice + augm_obs[1][i:]), ) + augm_obs[2:]
            if i < self.act_buf_size:  # we don't compute the action for the last observation of the trajectory
                new_action_distribution = self.model.actor(augm_obs)
                # this is stored in right -> left order for replacing correctly in augm_obs:
                self.traj_new_actions[self.act_buf_size -
                                      i - 1] = new_action_distribution.rsample()
                self.traj_new_actions_detach[self.act_buf_size - i -
                                             1] = self.traj_new_actions[self.act_buf_size - i - 1].detach()
                # this is stored in left -> right order for to be consistent with the reward trajectory:
                self.traj_new_actions_log_prob[i] = new_action_distribution.log_prob(
                    self.traj_new_actions[self.act_buf_size - i - 1])
                self.traj_new_actions_log_prob_detach[i] = self.traj_new_actions_log_prob[i].detach(
                )
            # this is stored in left -> right order:
            self.traj_new_augm_obs[i] = augm_obs

        # We now compute the state-value estimate
        # (this can be a different position in the trajectory for each element of the batch).
        # We expect each augmented state to be of shape (obs:tensor, act_buf:(tensor, ..., tensor), obs_del:tensor, act_del:tensor). Each tensor is batched.
        # To execute only 1 forward pass in the state-value estimator we recreate an artificially batched augmented state for this specific purpose.

        # FIXME: the profiler indicates that the following 5 lines are very inefficient, optimize

        obs_s = T.stack([self.traj_new_augm_obs[i + 1][0][ibatch]
                        for ibatch, i in enumerate(nstep_len)])
        act_s = tuple(T.stack([self.traj_new_augm_obs[i + 1][1][iact][ibatch]
                      for ibatch, i in enumerate(nstep_len)]) for iact in range(self.old_act_buf_size))
        od_s = T.stack([self.traj_new_augm_obs[i + 1][2][ibatch]
                       for ibatch, i in enumerate(nstep_len)])
        ad_s = T.stack([self.traj_new_augm_obs[i + 1][3][ibatch]
                       for ibatch, i in enumerate(nstep_len)])
        mod_augm_obs = tuple((obs_s, act_s, od_s, ad_s))

        with T.no_grad():

            # These are the delayed state-value estimates we are looking for:
            target_mod_val = [c(mod_augm_obs)
                              for c in self.model_target.critics]
            target_mod_val = reduce(T.min, T.stack(
                target_mod_val)).squeeze()  # minimum target estimate
            target_mod_val = target_mod_val * (1. - terminals)

            # Now let us use this to compute the state-value targets of the batch of initial augmented states:

            value_target = T.zeros(batch_size, device=self.device)
            backup_started = T.zeros(batch_size, device=self.device)
            for i in reversed(range(nstep_max_len + 1)):
                start_backup_mask = nstep_one_hot[:, i]
                backup_started += start_backup_mask
                value_target = self.reward_reward_scale * rew_traj[i] - self.entropy_reward_scale * self.traj_new_actions_log_prob_detach[i] + \
                    backup_started * self.discount * \
                    (value_target + start_backup_mask * target_mod_val)

        assert values[0].shape == value_target.shape, f"values[0].shape : {values[0].shape} != value_target.shape : {value_target.shape}"
        assert not value_target.requires_grad

        # Now the critic loss is:

        loss_critic = sum(mse_loss(v, value_target) for v in values)

        # actor loss:
        # TODO: there is probably a way of merging this with the previous for loop

        model_mod_val = [c(mod_augm_obs) for c in self.model_nograd.critics]
        model_mod_val = reduce(T.min, T.stack(
            model_mod_val)).squeeze()  # minimum model estimate
        model_mod_val = model_mod_val * (1. - terminals)

        loss_actor = T.zeros(batch_size, device=self.device)
        backup_started = T.zeros(batch_size, device=self.device)
        for i in reversed(range(nstep_max_len + 1)):
            start_backup_mask = nstep_one_hot[:, i]
            backup_started += start_backup_mask
            loss_actor = - self.entropy_reward_scale * \
                self.traj_new_actions_log_prob[i] + backup_started * self.discount * (
                    loss_actor + start_backup_mask * model_mod_val)
        loss_actor = - loss_actor.mean(0)

        # update model
        self.optimizer.zero_grad()
        loss_total = self.loss_alpha * loss_actor + \
            (1 - self.loss_alpha) * loss_critic
        loss_total.backward()
        self.optimizer.step()

        # update target model
        exponential_moving_average(self.model_target.parameters(
        ), self.model.parameters(), self.target_update)

        # exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.target_update)  # this is for trying PopArt in the future

        return dict(
            loss_total=loss_total.detach(),
            loss_critic=loss_critic.detach(),
            loss_actor=loss_actor.detach(),
            memory_size=len(self.memory),)
