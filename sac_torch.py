import os
import sys
import warnings
import pickle
import numpy as np
from copy import deepcopy

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nn import no_grad, exponential_moving_average

from buffer import ReplayBuffer
from buffer import TrajMemoryNoHidden
from networks import ActorNetwork, CriticNetwork, ValueNetwork, ActorLSTM, ValueNetworkLSTM, CriticNetworkLSTM

from models.LSTM import LSTMModel
from utils.LSTM_model_optimizer import Optimization

from dcac_models import Mlp
from functools import reduce
from torch.nn.functional import mse_loss


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
                 env_id, gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=256,
                 layer2_size=256, batch_size=100, reward_scale=2, init_policy='',
                 actor_net='', device='cuda', agent_args=None):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.init_policy = init_policy
        self.device = device
        self.agent_args = agent_args
        self.actor_net = actor_net

        if actor_net == 'LSTM':
            ARGS_PATH = './policy_args/'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                with open(ARGS_PATH + init_policy + '/args.pkl', 'rb') as file:
                    model_args = pickle.load(file)

            model = LSTMModel(model_args, self.device).float()
            try:
                model.load_state_dict(
                    T.load(os.path.join('./policy_checkpoints/' + init_policy, 'checkpoint.pth')))
            except OSError as e:
                print(
                    f"Unable to open {self.checkpoint_file}: {e}", file=sys.stderr)
                return

            loss_fn = nn.MSELoss(reduction="mean")
            optimizer = optim.Adam(model.parameters(), lr=model_args.learning_rate,
                                   weight_decay=model_args.weight_decay)

            opt = Optimization(model=model, loss_fn=loss_fn,
                               optimizer=optimizer, args=model_args,
                               setting=model_args.setting, device=self.device,
                               target_idx=0, is_policy=True)

            self.actor = ActorLSTM(name=env_id+'_actor_LSTM', min_action=env.action_space.low,
                                   max_action=env.action_space.high,
                                   model=model, opt=opt, model_args=model_args,
                                   lstm_output_dims=len(model_args.ctrl_vars),
                                   n_actions=n_actions, init_policy=self.init_policy)

            self.critic_1 = CriticNetworkLSTM(beta, input_dims, layer1_size,
                                              layer2_size, n_actions=n_actions,
                                              name=env_id+'_critic_1',
                                              init_policy=self.init_policy)

            self.critic_2 = CriticNetworkLSTM(beta, input_dims, layer1_size,
                                              layer2_size, n_actions=n_actions,
                                              name=env_id+'_critic_2',
                                              init_policy=self.init_policy)

            self.value = ValueNetworkLSTM(beta, input_dims, layer1_size,
                                          layer2_size, name=env_id+'_value',
                                          init_policy=self.init_policy)

            self.target_value = ValueNetworkLSTM(beta, input_dims, layer1_size,
                                                 layer2_size, name=env_id+'_target_value',
                                                 init_policy=self.init_policy)

        else:
            self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name=env_id+'_actor',
                                      max_action=env.action_space.high)

            self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                          layer2_size, n_actions=n_actions,
                                          name=env_id+'_critic_1',
                                          init_policy=self.init_policy)

            self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                          layer2_size, n_actions=n_actions,
                                          name=env_id+'_critic_2',
                                          init_policy=self.init_policy)

            self.value = ValueNetwork(beta, input_dims, layer1_size,
                                      layer2_size, name=env_id+'_value',
                                      init_policy=self.init_policy)

            self.target_value = ValueNetwork(beta, input_dims, layer1_size,
                                             layer2_size, name=env_id+'_target_value',
                                             init_policy=self.init_policy)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

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

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        # Value loss
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(
            state, reparameterize=False)
        # actions, log_probs = self.actor.sample_mvnormal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)

        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * (F.mse_loss(value, value_target))
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Actor loss
        actions, log_probs = self.actor.sample_normal(
            state, reparameterize=True)
        # actions, log_probs = self.actor.sample_mvnormal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)

        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)

        if self.actor_net == 'LSTM':
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            # Gradient clipping to prevent exploding gradient problem
            T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor.optimizer.step()
        else:
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
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

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

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


class RandomDelayAgent():
    def __init__(self, batchsize=256, memory_size=1000000, lr=0.0003, discount=0.99, target_update=0.005, 
                 reward_scale=5., entropy_scale=1., start_training=10000, device=None, training_steps=1):
        super(RandomDelayAgent, self).__init__()
        Model: type = Mlp
        loss_alpha: float = 0.2
        rtac: bool = False

    def __post_init__(self, Env):
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space
            self.sup_obs_delay = env.obs_delay_range.stop
            self.sup_act_delay = env.act_delay_range.stop
            self.act_buf_size = self.sup_obs_delay + self.sup_act_delay - 1
            self.old_act_buf_size = deepcopy(self.act_buf_size)
            if self.rtac:
                self.act_buf_size = 1

        assert self.device is not None
        # or ("cuda" if torch.cuda.is_available() else "cpu")
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
            self.memory_size, self.batchsize, device, history=self.act_buf_size)

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
                value_target = self.reward_scale * rew_traj[i] - self.entropy_scale * self.traj_new_actions_log_prob_detach[i] + \
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
            loss_actor = - self.entropy_scale * \
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
