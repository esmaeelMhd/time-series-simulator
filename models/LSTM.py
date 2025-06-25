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
import torch.nn.functional as F

#%%

class LSTMModel(nn.Module):    
    def __init__(self, args):
        """The __init__ method that initiates an LSTM instance.

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
        
        # Setup the device based on GPU usage
        self.device = torch.device('cuda') if torch.cuda.is_available() and self.args.use_gpu else torch.device('cpu')
        if self.device == 'cuda':
            self.device_ids = ','.join(str(i) for i in range(torch.cuda.device_count()))
            self.use_multi_gpu = torch.cuda.device_count() > 1
        else:
            self.device_ids = ''
            self.use_multi_gpu = False  
            
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.args.in_features, 
            hidden_size=self.args.hidden_dim, 
            num_layers=self.args.layer_dim, 
            dropout=self.args.dropout,
            batch_first=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(
            in_features=self.args.seq_len * self.args.hidden_dim, 
            out_features=self.args.pred_len * self.args.out_features,  # Adjust the output shape for multiple predictions
            bias=True
        )

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, pred_len, output_dim)

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
        
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        batch_size = x.shape[0]
        out = out.contiguous().view(batch_size, -1)
        out = out.to(x.device)
        
        # Convert the final state to our desired output shape (batch_size, pred_len, output_dim)
        out = self.fc(out)
        out = out.view(x.size(0), self.args.pred_len, self.args.out_features)  # Reshape for multiple predictions
                
        return out

#%%
    
class EncoderLSTM(torch.nn.Module):
    def __init__(self, args):
        super(EncoderLSTM, self).__init__()  
        self.args = args        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.args.in_features, 
            hidden_size=self.args.hidden_dim, 
            num_layers=self.args.layer_dim, 
            dropout=self.args.dropout,
            batch_first=True,
        )

    def forward(self, input, hidden): # input [batch_size, length T, dimensionality d]      
        output, hidden = self.lstm(input, hidden)      
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        # [num_layers*num_directions, batch, hidden_size]   
        return (torch.zeros(self.args.layer_dim, batch_size, self.args.hidden_dim, device=device),
                torch.zeros(self.args.layer_dim, batch_size, self.args.hidden_dim, device=device))
    
class DecoderLSTM(nn.Module):
    def __init__(self, args):
        super(DecoderLSTM, self).__init__()  
        self.args = args
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.args.in_features, 
            hidden_size=self.args.hidden_dim, 
            num_layers=self.args.layer_dim, 
            dropout=self.args.dropout,
            batch_first=True,
        )  
        
        self.fc = nn.Linear(self.args.hidden_dim, self.args.d_ff)
        self.out = nn.Linear(self.args.d_ff, self.args.in_features)   
        # self.out = nn.Linear(self.args.d_ff, self.args.in_features)  
        
    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden) 
        output = F.relu( self.fc(output))
        output = self.out(output)      
        return output, hidden
    
class Net_LSTM(nn.Module):
    def __init__(self, encoder, decoder, args, device):
        super(Net_LSTM, self).__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = self.args.pred_len
        self.device = device
        
    def forward(self, x):
        # print('input in Net shape: ', x.shape)
        batch_size = x.shape[0]
        input_length  = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(batch_size, self.device)
        # print('encoder hidden: ', encoder_hidden[0].shape)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:, ei:ei+1, :], encoder_hidden)
            
        # print('encoder output: ', encoder_output.shape)
        decoder_input = x[:, -1, :].unsqueeze(1) # first decoder input = last element of input sequence
        decoder_hidden = encoder_hidden
        
        outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]).to(self.device)
        # outputs = torch.zeros([x.shape[0], self.target_length, self.args.out_features]).to(self.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # print('decoder output: ', decoder_output.shape)
            decoder_input = decoder_output
            outputs[:, di:di+1, :] = decoder_output
        return outputs[:, :, :self.args.out_features]
