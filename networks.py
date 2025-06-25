import os
import sys
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

class ActorNetwork(nn.Module):
    """
    A unified actor network class that supports both LSTM-based and standard fully connected
    layers for action output in reinforcement learning depending on the specified configuration.

    Attributes:
        lr_actor (float): Learning rate for the actor.
        input_dims (tuple): Input dimensions to the network.
        fc1_dims (int): Number of neurons in the first fully connected layer.
        fc2_dims (int): Number of neurons in the second fully connected layer.
        n_actions (int): Number of possible actions.
        min_action (float): Minimum value for actions.
        max_action (float): Maximum value for actions.
        name (str): Name of the network, used for saving/loading checkpoints.
        chkpt_dir (str): Directory where checkpoints are stored.
        init_policy (str): Initial policy name, affects the checkpoint directory.
        use_lstm (bool): Flag indicating whether to use an LSTM layer.
        lstm_output_dims (int): Number of output dimensions from the LSTM layer.
        model (nn.Module): The pre-trained model used when LSTM is active.
        reparam_noise (float): Small noise value added for numerical stability in reparameterization.
        
    Methods:
        forward(state, hidden_state=None): Computes the action distributions based on the state.
        sample_normal(state, hidden_state=None, reparameterize=True): Samples actions using a normal distribution.
        save_checkpoint(): Saves the model's state dict to a checkpoint file.
        load_checkpoint(): Attempts to load the model's state dict from a checkpoint file.
    """
    def __init__(self, lr_actor, input_dims, fc1_dims, fc2_dims, min_action, max_action,
                 n_actions, name, chkpt_dir='./policy_checkpoints/', init_policy='',
                 use_lstm=False, lstm_output_dims=None, model=None):
        
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.min_action = min_action
        self.max_action = max_action
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir + init_policy
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.use_lstm = use_lstm
        self.model = model
        self.reparam_noise = 1e-6
        self.in_features = self.input_dims[0] * self.input_dims[1] if len(self.input_dims) == 2 \
            else self.input_dims[0]
        
        if self.use_lstm and model:
            self.fc_mu = nn.Linear(lstm_output_dims, n_actions)
            self.fc_sigma = nn.Linear(lstm_output_dims, n_actions)
        else:
            self.fc1 = nn.Linear(self.in_features, self.fc1_dims * n_actions)
            self.fc2 = nn.Linear(self.fc1_dims * n_actions, self.fc2_dims)
            self.mu = nn.Linear(self.fc2_dims, n_actions)
            self.sigma = nn.Linear(self.fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, hidden_state=None):
        if self.use_lstm and self.model:
            if state.ndim == 2:
                state = state.unsqueeze(1) 
            lstm_output = self.model(state).view(-1, self.n_actions)
            lstm_output = F.relu(lstm_output)
            mu = self.fc_mu(lstm_output)
            sigma = nn.functional.softplus(self.fc_sigma(lstm_output)) + self.reparam_noise
        else:
            # Assuming x is of shape [batch_size, seq_len, in_features]
            batch_size = state.shape[0] 
            # seq_len = state.shape[1]
            # Flatten the sequence
            state = state.view(batch_size, -1)
            
            prob = F.relu(self.fc1(state))
            prob = F.relu(self.fc2(prob))
            mu = self.mu(prob).view(batch_size, 1, self.n_actions)
            sigma = T.clamp(self.sigma(prob), min=self.reparam_noise, max=1)
            sigma = sigma.view(batch_size, 1, self.n_actions)

        return mu, sigma

    def sample_normal(self, state, hidden_state=None, reparameterize=True):
        mu, sigma = self.forward(state, hidden_state)
        probabilities = T.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        tanh_output = T.tanh(actions)
        action = T.tensor(self.min_action).to(self.device) +\
            (tanh_output + 1) * (T.tensor(self.max_action).to(self.device) -\
             T.tensor(self.min_action).to(self.device)) / 2 
                
        log_probs = probabilities.log_prob(actions)
        def safe_log(x):
            # Safe logarithm implementation
            eps = 1e-6
            return T.log(T.clamp(x, min=eps))
        
        log_probs -= safe_log(1 - action.pow(2) + self.reparam_noise)
        # log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        # FIXME: check if it is correct to sum all actions log_probs
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

class CriticNetwork(nn.Module):
    """
    A unified critic network class that can operate with either LSTM-based or standard fully connected
    layers for action-value estimation in reinforcement learning, depending on the specified configuration.

    Attributes:
        lr_critic (float): Learning rate for the critic.
        input_dims (tuple): Input dimensions to the network.
        fc1_dims (int): Number of neurons in the first fully connected layer.
        fc2_dims (int): Number of neurons in the second fully connected layer.
        n_actions (int): Number of possible actions.
        name (str): Name of the network, used for saving/loading checkpoints.
        chkpt_dir (str): Directory where checkpoints are stored.
        init_policy (str): Initial policy name, affects the checkpoint directory.
        use_lstm (bool): Flag indicating whether to use LSTM layers.
        lstm_hidden_dim (int): Number of units in the LSTM layer if used.

    Methods:
        forward(state, action): Computes the Q-value based on the state and action.
        save_checkpoint(): Saves the model's state dict to a checkpoint file.
        load_checkpoint(): Attempts to load the model's state dict from a checkpoint file.
    """
    def __init__(self, lr_critic, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='./policy_checkpoints/', init_policy='', use_lstm=False,
                 lstm_hidden_dim=256):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir + init_policy
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.use_lstm = use_lstm
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_obs = self.input_dims[1] if len(self.input_dims) == 2 else self.input_dims[0]

        if self.use_lstm:
            # Set up the LSTM layer for processing sequences
            self.lstm = nn.LSTM(input_size=self.input_dims[1],
                                hidden_size=self.lstm_hidden_dim,
                                batch_first=True)
            input_to_fc1 = self.lstm_hidden_dim + n_actions
        else:
            # Direct input to the first fully connected layer
            input_to_fc1 = self.n_obs + n_actions

        # Fully connected layers
        self.fc1 = nn.Linear(input_to_fc1, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        if self.use_lstm:
            # Process the input through the LSTM layer if LSTM is used
            lstm_out, _ = self.lstm(state)
            lstm_out = lstm_out[:, -1, :]  # Taking the output of the last time step
            combined_input = T.cat([lstm_out, action], dim=1)
        else:
            # Concatenate state and action directly if no LSTM is used
            batch_size = state.shape[0]
            state = state[:, -1, :].view(batch_size, -1)
            action = action.view(batch_size, -1)
            combined_input = T.cat([state, action], dim=1)

        action_value = F.relu(self.fc1(combined_input))
        action_value = F.relu(self.fc2(action_value))
        q1 = self.q1(action_value)
        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        try:
            self.load_state_dict(T.load(self.checkpoint_file))
        except OSError as e:
            print(f"Unable to open {self.checkpoint_file}: {e}", file=sys.stderr)
            return

class ValueNetwork(nn.Module):
    """
    A unified class for a value network that can operate either as a standard fully connected network or
    incorporate an LSTM layer for handling sequential data, based on the initialization parameters.

    Attributes:
        lr_critic (float): Learning rate for the critic.
        input_dims (tuple): Input dimensions to the network.
        fc1_dims (int): Number of neurons in the first fully connected layer.
        fc2_dims (int): Number of neurons in the second fully connected layer.
        name (str): Name of the network, used for checkpointing.
        chkpt_dir (str): Directory where checkpoints are stored.
        init_policy (str): Initial policy name for checkpointing.
        use_lstm (bool): Flag to determine whether to use an LSTM layer.
        lstm_hidden_dim (int): Number of hidden units in the LSTM layer if used.

    Methods:
        forward(state): Defines the computation performed at every call.
        save_checkpoint(): Saves the model's state dict to a checkpoint file.
        load_checkpoint(): Loads the model's state dict from a checkpoint file.
    """
    def __init__(self, lr_critic, input_dims, fc1_dims, fc2_dims, name,
                 chkpt_dir='./policy_checkpoints/', init_policy='', use_lstm=False,
                 lstm_hidden_dim=256):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir + init_policy
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.use_lstm = use_lstm
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_obs = self.input_dims[1] if len(self.input_dims) == 2 else self.input_dims[0]
        self.seq_len = self.input_dims[0] if len(self.input_dims) == 2 else 1

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=self.n_obs,
                                hidden_size=self.lstm_hidden_dim,
                                batch_first=True)
            self.fc1 = nn.Linear(lstm_hidden_dim, self.fc1_dims)
        else:
            self.fc1 = nn.Linear(self.n_obs, self.fc1_dims)
            
        # Fully connected layers
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # state shape is expected to be (batch_size, sequence_length, number_of_features)
        state = state.view(-1, self.seq_len, self.input_dims[-1])
        if self.use_lstm:
            lstm_out, _ = self.lstm(state)
            # Take the output of the last time step
            last_time_step = lstm_out[:, -1, :]
            state_value = F.relu(self.fc1(last_time_step))
        else:
            batch_size = state.shape[0] 
            state = state[:, -1, :].view(batch_size, -1)
            state_value = F.relu(self.fc1(state))

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