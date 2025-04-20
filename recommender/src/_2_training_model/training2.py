import torch
from torch_geometric.data import DataLoader
from torch.optim import Adam
from _3_evaluating_model.evaluate_model import link_prediction_loss
from _1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0

def train(model, train_data, neg_train_data, optimizer):
    model.train()
    optimizer.zero_grad()

    # Forward pass: Get embeddings for the training data
    z = model(train_data.x, train_data.edge_index)

    # Prepare negative samples for link prediction
    neg_edge_index = neg_train_data  # Negative edges for training

    # Compute the loss (link prediction)
    loss = link_prediction_loss(train_data.edge_index, neg_edge_index, z)

    # Backpropagation
    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate(model, val_data, neg_val_data):
    model.eval()

    with torch.no_grad():
        # Get embeddings for the validation data
        z = model(val_data.x, val_data.edge_index)
        
        # Prepare negative edges for evaluation
        neg_edge_index = neg_val_data  # Negative edges for validation
        
        # Compute the validation loss (link prediction)
        loss = link_prediction_loss(val_data.edge_index, neg_edge_index, z)
    
    return loss.item()

# Hyperparameters
in_channels = 128  # Adjust based on your data
hidden_channels = 64
out_channels = 32
lr = 0.01

# Initialize the model and optimizer
model = GraphSAGEModelV0(in_channels, hidden_channels, out_channels)
optimizer = Adam(model.parameters(), lr=lr)

# Load the data (ensure correct paths and formats)
train_data = torch.load('data/processed/train_data.pt')  # Example loading
neg_train_data = torch.load('data/processed/train_neg_samples.pt')      

val_data = torch.load('data/processed/val_data.pt')  # Example loading
neg_val_data = torch.load('data/processed/val_neg_samples.pt') 

# Training loop
epochs = 100
for epoch in range(epochs):
    # Train on your training data
    loss = train(model, train_data, neg_train_data, optimizer)
    
    # Evaluate on validation data
    val_loss = evaluate(model, val_data, neg_val_data)

    # Print loss and validation loss for each epoch
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
