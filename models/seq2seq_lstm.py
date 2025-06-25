import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def init_hidden(self, device):
        # [num_layers*num_directions, batch, hidden_size]   
        return torch.zeros(self.args.layer_dim, self.args.batch_size, self.args.hidden_dim, device=device)
    
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
        self.out = nn.Linear(self.args.d_ff, self.args.out_features)         
        
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
        input_length  = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:, ei:ei+1, :], encoder_hidden)
            
        decoder_input = x[:, -1, :].unsqueeze(1) # first decoder input = last element of input sequence
        decoder_hidden = encoder_hidden
        
        outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]).to(self.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output
            outputs[:, di:di+1, :] = decoder_output
        return outputs      