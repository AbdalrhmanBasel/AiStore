"""
Test Suite for GraphSAGE Model
==============================
This file contains unit tests for the GraphSAGE model and its components.
It ensures that the model behaves correctly during forward passes, 
training steps, serialization, and on edge cases.

Testing Principles Followed:
- Isolation: Each test verifies a single behavior
- Coverage: Includes forward, backward, edge cases, and serialization
- Robustness: Checks for invalid inputs and TorchScript compatibility
"""

import torch
import pytest
from src.models.graphsage import GraphSAGE
from src.models.encoders import Encoder


# --------- Constants --------- #
BATCH_SIZE  = 4
FEATURE_DIM = 16
EDGE_DIM    = 4
NUM_CLASSES = 5
NUM_NODES   = 10


# --------- Fixtures --------- #
@pytest.fixture
def default_model():
    return GraphSAGE(input_dim=FEATURE_DIM, hidden_dim=32, output_dim=NUM_CLASSES)

@pytest.fixture
def default_encoder():
    return Encoder(input_dim=FEATURE_DIM, hidden_dim=32)

@pytest.fixture
def dummy_graph():
    x = torch.randn(BATCH_SIZE, FEATURE_DIM)
    edge_index = torch.randint(0, BATCH_SIZE, (2, EDGE_DIM))
    return x, edge_index


# --------- Forward Pass Tests --------- #

def test_forward_pass_output_shape_matches_num_classes(default_model, dummy_graph):
    x, edge_index = dummy_graph
    out = default_model(x, edge_index)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES), \
        f"Expected output shape {(BATCH_SIZE, NUM_CLASSES)}, got {out.shape}"


def test_forward_pass_on_small_graph(default_model):
    x = torch.randn(NUM_NODES, FEATURE_DIM)
    edge_index = torch.randint(0, NUM_NODES, (2, EDGE_DIM))
    out = default_model(x, edge_index)
    assert out.shape == (NUM_NODES, NUM_CLASSES), \
        f"Expected output shape {(NUM_NODES, NUM_CLASSES)}, got {out.shape}"


def test_forward_pass_on_empty_graph(default_model):
    x = torch.randn(0, FEATURE_DIM)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    out = default_model(x, edge_index)
    assert out.shape == (0, NUM_CLASSES), \
        f"Expected output shape (0, NUM_CLASSES), got {out.shape}"


def test_forward_pass_invalid_input_dimension(default_model):
    x_invalid = torch.randn(BATCH_SIZE, FEATURE_DIM + 1)
    edge_index = torch.randint(0, BATCH_SIZE, (2, EDGE_DIM))
    with pytest.raises((RuntimeError, ValueError)):
        default_model(x_invalid, edge_index)


# --------- Encoder Tests --------- #

def test_encoder_output_dimensions_consistent_with_hidden_size(default_encoder):
    x = torch.randn(BATCH_SIZE, FEATURE_DIM)
    encoded = default_encoder(x)
    assert encoded.shape == (BATCH_SIZE, 32), \
        f"Expected encoded shape (BATCH_SIZE, 32), got {encoded.shape}"


# --------- Training Tests --------- #

def test_training_step_updates_weights(default_model, dummy_graph):
    x, edge_index = dummy_graph
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    optimizer = torch.optim.Adam(default_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    output = default_model(x, edge_index)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, f"Loss should be > 0, got {loss.item()}"


def test_training_step_with_sgd_optimizer(default_model, dummy_graph):
    x, edge_index = dummy_graph
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    optimizer = torch.optim.SGD(default_model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    output = default_model(x, edge_index)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, f"Loss should be > 0 with SGD optimizer, got {loss.item()}"


def test_gradient_flow_through_all_layers(default_model, dummy_graph):
    x, edge_index = dummy_graph
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    criterion = torch.nn.CrossEntropyLoss()
    output = default_model(x, edge_index)
    loss = criterion(output, y)

    default_model.zero_grad()
    loss.backward()

    for name, param in default_model.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"


def test_model_overfits_tiny_dataset():
    model = GraphSAGE(input_dim=FEATURE_DIM, hidden_dim=32, output_dim=NUM_CLASSES)
    x = torch.randn(2, FEATURE_DIM)
    edge_index = torch.randint(0, 2, (2, EDGE_DIM))
    y = torch.tensor([0, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(100):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1, f"Expected loss to drop below 0.1 after overfitting, got {loss.item()}"


# --------- Serialization Tests --------- #

def test_model_saving_and_loading_preserves_weights(tmp_path, default_model):
    path = tmp_path / "model.pth"
    torch.save(default_model.state_dict(), path)

    loaded_model = GraphSAGE(input_dim=FEATURE_DIM, hidden_dim=32, output_dim=NUM_CLASSES)
    loaded_model.load_state_dict(torch.load(path))

    for p1, p2 in zip(default_model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2), "Mismatch in model weights after loading"


# --------- Compatibility Tests --------- #

def test_model_torchscript_compatibility(default_model, dummy_graph):
    x, edge_index = dummy_graph
    try:
        scripted_model = torch.jit.script(default_model)
        out = scripted_model(x, edge_index)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)
    except Exception as e:
        pytest.fail(f"TorchScript compatibility failed: {e}")
