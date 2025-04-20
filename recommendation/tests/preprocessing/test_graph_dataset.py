import torch
import pytest
from torch_geometric.data import DataLoader
from recommendation.src.datasets.LinkPredictionDataset import GraphDataset

NUM_NODES = 100
NUM_FEATURES = 16
NUM_CLASSES = 5

@pytest.fixture
def dummy_graph_dataset(tmp_path):
    """Creates a small dummy graph and saves it as .pt files, simulating preprocessed graph storage."""
    x = torch.randn(NUM_NODES, NUM_FEATURES)
    y = torch.randint(0, NUM_CLASSES, (NUM_NODES,))
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_NODES * 2))

    torch.save(x, tmp_path / "features.pt")
    torch.save(y, tmp_path / "labels.pt")
    torch.save(edge_index, tmp_path / "edge_index.pt")

    return GraphDataset(
        root_dir=tmp_path,
        feature_file="features.pt",
        label_file="labels.pt",
        edge_index_file="edge_index.pt"
    )

def test_dataset_length(dummy_graph_dataset):
    assert len(dummy_graph_dataset) == NUM_NODES, f"Expected {NUM_NODES} samples, got {len(dummy_graph_dataset)}"

def test_feature_shape(dummy_graph_dataset):
    data = dummy_graph_dataset[0]
    assert data.x.shape[1] == NUM_FEATURES, f"Expected feature dim {NUM_FEATURES}, got {data.x.shape[1]}"

def test_label_type_and_shape(dummy_graph_dataset):
    data = dummy_graph_dataset[0]
    assert isinstance(data.y.item(), int), "Label should be an integer class"
    assert data.y.dim() == 0 or data.y.shape == torch.Size([]), f"Label should be a scalar, got shape {data.y.shape}"

def test_edge_index_shape(dummy_graph_dataset):
    data = dummy_graph_dataset[0]
    assert data.edge_index.shape[0] == 2, f"Expected edge_index to have shape (2, E), got {data.edge_index.shape}"

def test_node_indices_within_bounds(dummy_graph_dataset):
    data = dummy_graph_dataset[0]
    max_index = data.x.shape[0] - 1
    assert torch.all(data.edge_index < data.x.shape[0]), "Edge index contains invalid node references"

def test_empty_graph_support(tmp_path):
    torch.save(torch.empty(0, NUM_FEATURES), tmp_path / "features.pt")
    torch.save(torch.empty(0, dtype=torch.long), tmp_path / "labels.pt")
    torch.save(torch.empty(2, 0, dtype=torch.long), tmp_path / "edge_index.pt")

    dataset = GraphDataset(
        root_dir=tmp_path,
        feature_file="features.pt",
        label_file="labels.pt",
        edge_index_file="edge_index.pt"
    )

    assert len(dataset) == 0, f"Expected empty dataset, got {len(dataset)}"

def test_incorrect_feature_file_raises_error(tmp_path):
    torch.save(torch.randn(5, NUM_FEATURES), tmp_path / "features.pt")
    torch.save(torch.randint(0, NUM_CLASSES, (10,)), tmp_path / "labels.pt")  # mismatch
    torch.save(torch.randint(0, 10, (2, 20)), tmp_path / "edge_index.pt")

    with pytest.raises(ValueError):
        _ = GraphDataset(
            root_dir=tmp_path,
            feature_file="features.pt",
            label_file="labels.pt",
            edge_index_file="edge_index.pt"
        )

def test_dataset_batching(dummy_graph_dataset):
    loader = DataLoader(dummy_graph_dataset, batch_size=8, shuffle=False)
    for batch in loader:
        assert hasattr(batch, 'x'), "Batch must have node features"
        assert hasattr(batch, 'y'), "Batch must have labels"
        assert hasattr(batch, 'edge_index'), "Batch must have edge_index"
        assert batch.x.size(0) >= 1, "Each batch must contain nodes"

def test_dataloader_batch_shapes(dummy_graph_dataset):
    loader = DataLoader(dummy_graph_dataset, batch_size=16, shuffle=False)
    batch = next(iter(loader))

    assert batch.x.dim() == 2, "Batch x should be 2D"
    assert batch.y.dim() == 1, "Batch y should be 1D"
    assert batch.edge_index.shape[0] == 2, "Edge index should have shape (2, E)"
