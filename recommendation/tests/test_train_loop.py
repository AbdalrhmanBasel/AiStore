import torch
import pytest
from src.models.graphsage import GraphSAGE
from src.train import train_loop

BATCH_SIZE = 8
FEATURE_DIM = 16
EDGE_DIM = 12
NUM_CLASSES = 4


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, index):
        return {
            'x': torch.randn(BATCH_SIZE, FEATURE_DIM),
            'edge_index': torch.randint(0, BATCH_SIZE, (2, EDGE_DIM)),
            'y': torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        }

    def __len__(self):
        return self.num_samples


def test_train_loop_parameter_update():
    """
    Test that model parameters are updated after one epoch of training.
    """
    dataset = DummyDataset(num_samples=10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = GraphSAGE(input_dim=FEATURE_DIM, hidden_dim=32, output_dim=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    initial_params = [param.clone() for param in model.parameters()]

    train_loop(model, dataloader, optimizer, criterion, device='cpu', epoch=1)

    # Ensure at least one parameter changed
    changed = any(not torch.equal(p0, p1) for p0, p1 in zip(initial_params, model.parameters()))
    assert changed, "Model parameters did not update during training."


def test_train_loop_no_crash_on_empty_batch():
    """
    Ensure that the training loop does not crash when given an empty batch.
    """

    class EmptyDataset(torch.utils.data.Dataset):
        def __getitem__(self, index):
            return {
                'x': torch.empty(0, FEATURE_DIM),
                'edge_index': torch.empty(2, 0, dtype=torch.long),
                'y': torch.empty(0, dtype=torch.long)
            }

        def __len__(self):
            return 1

    dataset = EmptyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = GraphSAGE(input_dim=FEATURE_DIM, hidden_dim=32, output_dim=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    try:
        train_loop(model, dataloader, optimizer, criterion, device='cpu', epoch=1)
    except Exception as e:
        pytest.fail(f"Training loop crashed on empty batch: {e}")


def test_train_loop_loss_decreasing():
    """
    Check if the training loss decreases over multiple epochs.
    """
    dataset = DummyDataset(num_samples=20)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = GraphSAGE(input_dim=FEATURE_DIM, hidden_dim=32, output_dim=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []

    def capture_loss(model, loader, optimizer, criterion):
        total_loss = 0
        for batch in loader:
            x = batch['x']
            edge_index = batch['edge_index']
            y = batch['y']
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss

    for _ in range(3):
        loss = capture_loss(model, dataloader, optimizer, criterion)
        losses.append(loss)

    assert losses[1] < losses[0] or losses[2] < losses[1], "Loss did not decrease over epochs."


def test_train_loop_gradient_flow():
    """
    Ensure gradients flow during backprop.
    """
    dataset = DummyDataset(num_samples=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = GraphSAGE(input_dim=FEATURE_DIM, hidden_dim=32, output_dim=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        x = batch['x']
        edge_index = batch['edge_index']
        y = batch['y']
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient is None for parameter {name}"
        break
