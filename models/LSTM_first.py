"""
Created on Monday July 01 2022
@author: Esmaeel Mohammadi

# =============================================================================
# This script is used to create an instance of LSTM model:
    1. Creates the model based on inputs
    2. The purpose is sequence to sequence modelling
    3. Input sequence: [batch_size, sequence_length, n_features]
    4. Output sequence: [batch_size, prediction_length, n_features]
    5. Prediction length is currently ONE
# =============================================================================
"""

import torch
import torch.nn as nn

#%% LSTM Model
    
class LSTMModel(nn.Module):
    
    def __init__(self, args, device):
        """The __init__ method that initiates a LSTM instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(LSTMModel, self).__init__()

        # Defining args
        self.args = args
        self.device = device
        
        # LSTM layers
        self.lstm = nn.LSTM(
           input_size = self.args.in_features, 
           hidden_size = self.args.hidden_dim, 
           num_layers = self.args.layer_dim, 
           dropout = self.args.dropout,
           batch_first = True, 
        )

        # Fully connected layer
        self.fc = nn.Linear(in_features=self.args.seq_len * self.args.hidden_dim, out_features=self.args.out_features, bias=True)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """              
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.args.layer_dim, x.size(0), self.args.hidden_dim).requires_grad_().to(x.device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.args.layer_dim, x.size(0), self.args.hidden_dim).requires_grad_().to(x.device)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        self.lstm.flatten_parameters()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # print('Out shape Before reshape: ', out.shape)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        # out = out[:, -1, :].to(x.device)
        batch_size = x.shape[0]
        out = out.contiguous().view(batch_size,-1)
        out = out.to(x.device)
        # print('Out shape After reshape: ', out.shape)

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        # out = out.view(x.size(0), 1, self.args.out_features)
        # print('Out shape After FC: ', out.shape)
        
        return out