import torch
from torch.optim import Adam
from src._1_model_selection.GraphSAGEModelV1 import GraphSAGEModelV1
from src._0_data_preprocessing.preprocessing import load_data
from src._3_evaluating_model import evaluate_model

from settings import EDGE_INDEX_PATH, FEATURES_PATH, NEGATIVE_SAMPLES_PATH

def train():
    # Load preprocessed data
    train_data = torch.load(EDGE_INDEX_PATH)  # Load edge indices
    train_features = torch.load(FEATURES_PATH)  # Load node features
    neg_train_data = torch.load(NEGATIVE_SAMPLES_PATH)  # Negative sampling data

    # Initialize the model
    model = GraphSAGEModelV1(input_dim=train_features.shape[1], hidden_dim=64, output_dim=32)

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        # Training the model
        model.train()
        loss = train_step(model, train_data, neg_train_data, optimizer)

        # Evaluate the model
        val_loss = evaluate(model, val_data, neg_val_data)

        # Save model checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    train()
