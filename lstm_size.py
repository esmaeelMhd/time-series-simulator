import torch.nn as nn

num_layers = 2
lstm = nn.LSTM(input_size=14, hidden_size=256, num_layers=num_layers)

# Calculate the number of parameters
num_params = sum(p.numel() for p in lstm.parameters() if p.requires_grad)
print(f'The number of parameters in the model: {num_params}')

# Output the number of layers
print(f'The number of layers in the model: {lstm.num_layers}')