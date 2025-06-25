import os
import sys
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                 name, chkpt_dir='./policy_checkpoints/', init_policy=''):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir  + init_policy
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # I think this breaks if the env has a 2D state representation
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device) 

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        try:
            self.load_state_dict(T.load(self.checkpoint_file))
        except OSError as e:
            print(f"Unable to open {self.checkpoint_file}: {e}", file=sys.stderr)
            return

class CriticNetworkLSTM(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 lstm_hidden_dim=256, chkpt_dir='./policy_checkpoints/', init_policy=''):
        super(CriticNetworkLSTM, self).__init__()
        self.input_dims = input_dims  # Expected to be (sequence_length, number_of_features)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.lstm_hidden_dim = lstm_hidden_dim
        self.name = name
        self.checkpoint_dir = chkpt_dir + init_policy
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # LSTM layer for processing time series state
        self.lstm = nn.LSTM(input_size=self.input_dims[1], 
                            hidden_size=self.lstm_hidden_dim, 
                            batch_first=True)

        # Fully connected layers
        # The input dimension here is the output dimension of LSTM plus the number of actions
        self.fc1 = nn.Linear(self.lstm_hidden_dim + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        # LSTM layer expects input of shape (batch, seq_len, features)
        lstm_out, _ = self.lstm(state)
        # Taking the output of the last time step
        lstm_out = lstm_out[:, -1, :]
        
        action = action.view((-1, self.n_actions))
        # Combine the output of LSTM with action
        state_action_value = T.cat([lstm_out, action], dim=1)

        state_action_value = F.relu(self.fc1(state_action_value))
        state_action_value = F.relu(self.fc2(state_action_value))

        q1 = self.q1(state_action_value)

        return q1
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        try:
            self.load_state_dict(T.load(self.checkpoint_file))
        except OSError as e:
            print(f"Unable to open {self.checkpoint_file}: {e}", file=sys.stderr)
            return

        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, max_action,
            n_actions, name, chkpt_dir='./policy_checkpoints/', init_policy=''):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_action = max_action
        self.checkpoint_dir = chkpt_dir  + init_policy
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        #sigma = T.sigmoid(self.sigma(prob))
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1) 
        # authors use -20, 2 -> doesn't seem to work for my implementation

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = T.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample() # reparameterizes the policy
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device) 
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def sample_mvnormal(self, state, reparameterize=True):
        """
            Doesn't quite seem to work.  The agent never learns.
        """
        mu, sigma = self.forward(state)
        n_batches = sigma.size()[0]

        cov = [sigma[i] * T.eye(self.n_actions).to(self.device) for i in range(n_batches)]
        cov = T.stack(cov)
        probabilities = T.distributions.MultivariateNormal(mu, cov)

        if reparameterize:
            actions = probabilities.rsample() # reparameterizes the policy
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) # enforce the action bound for (-1, 1)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.sum(T.log(1-action.pow(2) + self.reparam_noise))
        log_probs = log_probs.sum(-1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        try:
            self.load_state_dict(T.load(self.checkpoint_file))
        except OSError as e:
            print(f"Unable to open {self.checkpoint_file}: {e}", file=sys.stderr)
            return

class ActorLSTM(nn.Module):
    def __init__(self, name, min_action, max_action, model, opt, model_args, lstm_output_dims, 
                 n_actions, chkpt_dir='./policy_checkpoints/', init_policy=''):
        super(ActorLSTM, self).__init__()
        # Define the layers
        self.name = name
        self.min_action = min_action
        self.max_action = max_action
        self.n_actions = n_actions
        self.checkpoint_dir = chkpt_dir + init_policy
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6
        self.model = model
        self.opt = opt
        
        self.fc_mu = nn.Linear(lstm_output_dims, self.n_actions)
        self.fc_sigma = nn.Linear(lstm_output_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=model_args.learning_rate)
        
        # Prevent vanishing gradient
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights in LSTM layers
                T.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden-hidden weights in LSTM layers
                T.nn.init.orthogonal_(param.data)
            elif 'bias' in name:  # Bias initialization
                param.data.fill_(0)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, hidden_state=None):
        # Forward pass through the pre-trained LSTM
        lstm_output = self.model(state).to(self.device).view(-1, self.n_actions)
        lstm_output = F.relu(lstm_output)
        
        # Compute the distribution of actions
        mu = self.fc_mu(lstm_output)
        # Ensure the standard deviation is positive; using softplus and add a small value for numerical stability
        sigma = nn.functional.softplus(self.fc_sigma(lstm_output)) + 1e-6
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1) 
        return mu, sigma
    
    def sample_normal(self, state, hidden_state=None, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = T.distributions.Normal(mu, sigma)
    
        if reparameterize:
            actions = probabilities.rsample()  # Reparameterization trick for stability
        else:
            actions = probabilities.sample()
    
        # Apply tanh to ensure actions are within the valid range
        # actions = T.tanh(actions)*T.tensor(self.max_action).to(self.device).view_as(actions)
        tanh_output = T.tanh(actions)
        action = T.tensor(self.min_action).to(self.device) +\
            (tanh_output + 1) * (T.tensor(self.max_action).to(self.device) -\
             T.tensor(self.min_action).to(self.device)) / 2

        log_probs = probabilities.log_prob(actions)
        def safe_log(x):
            # Safe logarithm implementation
            eps = 1e-6
            return T.log(T.clamp(x, min=eps))
        
        # In your method where you calculate log_probs
        log_probs -= safe_log(1 - action.pow(2) + self.reparam_noise)
        # log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
    
        return action, log_probs
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        try:
            self.load_state_dict(T.load(self.checkpoint_file))
        except OSError as e:
            print(f"Unable to open {self.checkpoint_file}: {e}", file=sys.stderr)
            return

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
            name, chkpt_dir='./policy_checkpoints/', init_policy=''):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir  + init_policy
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        try:
            self.load_state_dict(T.load(self.checkpoint_file))
        except OSError as e:
            print(f"Unable to open {self.checkpoint_file}: {e}", file=sys.stderr)
            return


class ValueNetworkLSTM(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, name,
                 lstm_hidden_dim=256, chkpt_dir='./policy_checkpoints/', init_policy=''):
        super(ValueNetworkLSTM, self).__init__()

        # input_dims is expected to be (sequence_length, number_of_features)
        self.sequence_length, self.number_of_features = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lstm_hidden_dim = lstm_hidden_dim
        self.name = name
        self.checkpoint_dir = chkpt_dir + init_policy
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.number_of_features, 
                            hidden_size=self.lstm_hidden_dim, 
                            batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(self.lstm_hidden_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # state shape is expected to be (batch_size, sequence_length, number_of_features)
        lstm_out, _ = self.lstm(state)
        # Take the output of the last time step
        last_time_step = lstm_out[:, -1, :]

        state_value = F.relu(self.fc1(last_time_step))
        state_value = F.relu(self.fc2(state_value))

        v = self.v(state_value)

        return v
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        try:
            self.load_state_dict(T.load(self.checkpoint_file))
        except OSError as e:
            print(f"Unable to open {self.checkpoint_file}: {e}", file=sys.stderr)
            return
