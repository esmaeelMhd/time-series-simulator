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
