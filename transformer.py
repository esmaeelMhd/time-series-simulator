import torch
import torch.nn as nn
import math

# Positional Encoding to add sequence order information
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Transformer-based Encoder-Decoder model
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        
        # Encoder and decoder input layers
        self.encoder_input_layer = nn.Linear(input_dim, d_model)
        self.decoder_input_layer = nn.Linear(output_dim, d_model)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(d_model=d_model, 
                                          nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout)

        # Output layer for regression task
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        # Apply input embeddings
        src = self.encoder_input_layer(src)  # (sequence_length, batch_size, d_model)
        tgt = self.decoder_input_layer(tgt)  # (sequence_length, batch_size, d_model)

        # Add positional encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Transformer forward pass
        output = self.transformer(src, tgt)

        # Final output layer
        output = self.output_layer(output)

        return output

# Example usage:
input_dim = 10  # Number of input features (for each time step in the sequence)
output_dim = 1  # Forecasting one step ahead
d_model = 64  # Dimension of embeddings
nhead = 4  # Number of attention heads
num_encoder_layers = 3  # Number of layers in the encoder
num_decoder_layers = 3  # Number of layers in the decoder
dim_feedforward = 128  # Feedforward network dimension
dropout = 0.1

model = TransformerTimeSeries(input_dim=input_dim, 
                              d_model=d_model, 
                              nhead=nhead, 
                              num_encoder_layers=num_encoder_layers, 
                              num_decoder_layers=num_decoder_layers, 
                              dim_feedforward=dim_feedforward, 
                              output_dim=output_dim, 
                              dropout=dropout)

# Example data
src = torch.rand(50, 32, input_dim)  # (sequence_length, batch_size, input_dim)
tgt = torch.rand(50, 32, output_dim)  # (sequence_length, batch_size, output_dim)

# Forward pass
output = model(src, tgt)
print(output.shape)  # Should return (sequence_length, batch_size, output_dim)


import torch.optim as optim

# Hyperparameters
lr = 1e-4
epochs = 10

# Optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Example batch of input and target data
    src = torch.rand(50, 32, input_dim)  # (sequence_length, batch_size, input_dim)
    tgt = torch.rand(50, 32, output_dim)  # (sequence_length, batch_size, output_dim)
    
    # Forward pass
    output = model(src, tgt)
    
    # Calculate loss
    loss = criterion(output, tgt)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
