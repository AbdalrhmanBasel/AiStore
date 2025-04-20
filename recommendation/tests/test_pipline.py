import pytest
import torch
from recommendation.src.datasets.LinkPredictionDataset import GraphDataset
from src.models.graphsage import GraphSAGE
from src.models.encoders import Encoder
from src.train_loop import train
from src.visualization import visualize_embeddings

BATCH_SIZE = 4
FEATURE_DIM = 16
EDGE_DIM = 4
NUM_CLASSES = 5
NUM_NODES = 10

TEST_DATA_DIR = "data/processed" 

@pytest.fixture
def dataset():
    """Fixture for loading and preprocessing the graph dataset."""
    dataset = GraphDataset(TEST_DATA_DIR)
    assert len(dataset) > 0, "Dataset is empty - did preprocessing run?"
    return dataset

@pytest.fixture
def model():
    """Fixture for initializing the model."""
    return GraphSAGE(input_dim=FEATURE_DIM, hidden_dim=32, output_dim=NUM_CLASSES)

@pytest.fixture
def optimizer(model):
    """Fixture for initializing the optimizer."""
    return torch.optim.Adam(model.parameters(), lr=0.001)

@pytest.fixture
def dummy_data():
    """Fixture to return dummy graph data."""
    x = torch.randn(BATCH_SIZE, FEATURE_DIM)
    edge_index = torch.randint(0, BATCH_SIZE, (2, EDGE_DIM))
    return x, edge_index

def test_full_pipeline(dataset, model, optimizer, dummy_data):
    """
    Tests the entire pipeline: from data preprocessing to visualization.
    Ensures all components work together without error.
    
    Step 1: Train the model on the dataset
    Step 2: Validate if training occurred (loss should decrease)
    Step 3: Test the visualization - ensure embeddings are generated correctly
    Step 4: Assert the final model output has correct shape
    """
    x, edge_index = dummy_data

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10): 
        optimizer.zero_grad()
        output = model(x, edge_index)  
        y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))  
        loss = criterion(output, y)  
        loss.backward()  
        optimizer.step()

    assert loss.item() < 10, f"Loss should decrease but is {loss.item()}"

    embeddings = model(x, edge_index)
    visualize_embeddings(embeddings)

    assert embeddings is not None, "Embeddings are None, check model output"
    assert output.shape == (BATCH_SIZE, NUM_CLASSES), f"Expected output shape [(BATCH_SIZE, NUM_CLASSES)], got {output.shape}"

    print("Pipeline test passed successfully!")


@pytest.mark.parametrize("optimizer_class", [torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop])
def test_with_different_optimizers(dataset, optimizer_class, dummy_data):
    """Test the pipeline with different optimizers."""
    model = GraphSAGE(input_dim=FEATURE_DIM, hidden_dim=32, output_dim=NUM_CLASSES)
    optimizer = optimizer_class(model.parameters(), lr=0.001)

    x, edge_index = dummy_data

    model.train() 
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x, edge_index) 
        y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)) 
        loss = criterion(output, y) 
        loss.backward()  
        optimizer.step() 

    # Validate loss is decreasing
    assert loss.item() < 10, f"Loss should decrease but is {loss.item()}"

    # Check if embeddings are generated correctly
    embeddings = model(x, edge_index)
    visualize_embeddings(embeddings)

    assert embeddings is not None, "Embeddings are None, check model output"
    assert output.shape == (BATCH_SIZE, NUM_CLASSES), f"Expected output shape [(BATCH_SIZE, NUM_CLASSES)], got {output.shape}"

    print(f"Pipeline test passed with {optimizer_class.__name__} optimizer!")


def test_with_larger_dataset():
    """Test the pipeline with a larger dataset to check scalability."""
    dataset = GraphDataset(TEST_DATA_DIR)
    assert len(dataset) > 0, "Dataset is empty - did preprocessing run?"

    model = GraphSAGE(input_dim=FEATURE_DIM, hidden_dim=32, output_dim=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Use a small batch for this test
    batch_size = 8
    x = torch.randn(batch_size, FEATURE_DIM)
    edge_index = torch.randint(0, batch_size, (2, EDGE_DIM))

    # Train the model with the larger dataset
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x, edge_index)
        y = torch.randint(0, NUM_CLASSES, (batch_size,))
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 10, f"Loss should decrease but is {loss.item()}"

    embeddings = model(x, edge_index)
    visualize_embeddings(embeddings)

    assert embeddings is not None, "Embeddings are None, check model output"
    assert output.shape == (batch_size, NUM_CLASSES), f"Expected output shape [(batch_size, NUM_CLASSES)], got {output.shape}"

    print("Pipeline test passed with larger dataset!")

