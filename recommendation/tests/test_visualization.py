import pytest
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for test environments
import matplotlib.pyplot as plt
from src.visualization import plot_embeddings

EMBEDDING_DIM = 16
NUM_NODES = 20
NUM_CLASSES = 3


@pytest.fixture
def dummy_embeddings_and_labels():
    embeddings = torch.randn(NUM_NODES, EMBEDDING_DIM)
    labels = torch.randint(0, NUM_CLASSES, (NUM_NODES,))
    return embeddings, labels


def test_plot_embeddings_runs_without_error(dummy_embeddings_and_labels):
    """
    Test that plot_embeddings runs and returns a matplotlib figure without error.
    """
    embeddings, labels = dummy_embeddings_and_labels
    fig = plot_embeddings(embeddings, labels)
    assert isinstance(fig, plt.Figure), "plot_embeddings did not return a matplotlib Figure."


def test_plot_embeddings_handles_empty_input():
    """
    Test that plot_embeddings raises an error for empty input.
    """
    embeddings = torch.empty(0, EMBEDDING_DIM)
    labels = torch.empty(0, dtype=torch.long)

    with pytest.raises(ValueError, match="Embeddings must have at least one sample"):
        plot_embeddings(embeddings, labels)


def test_plot_embeddings_input_shape_mismatch():
    """
    Ensure it raises if embeddings and labels length mismatch.
    """
    embeddings = torch.randn(10, EMBEDDING_DIM)
    labels = torch.randint(0, NUM_CLASSES, (5,))  # Wrong length

    with pytest.raises(ValueError, match="Mismatch between embeddings and labels"):
        plot_embeddings(embeddings, labels)


def test_plot_embeddings_dimension_reduction():
    """
    Ensure that high-dimensional embeddings are reduced to 2D.
    """
    embeddings = torch.randn(NUM_NODES, EMBEDDING_DIM)
    labels = torch.randint(0, NUM_CLASSES, (NUM_NODES,))
    fig = plot_embeddings(embeddings, labels)

    # Get axes and ensure it's a 2D scatter plot
    ax = fig.axes[0]
    paths = ax.collections[0].get_offsets()
    assert paths.shape[1] == 2, "Embeddings were not reduced to 2D for visualization."


def test_plot_embeddings_color_mapping_consistency():
    """
    Ensure each class has a consistent color.
    """
    embeddings = torch.randn(NUM_NODES, EMBEDDING_DIM)
    labels = torch.tensor([i % NUM_CLASSES for i in range(NUM_NODES)])
    fig = plot_embeddings(embeddings, labels)

    ax = fig.axes[0]
    scatter = ax.collections[0]
    colors = scatter.get_array()
    unique_colors = torch.unique(torch.tensor(colors)).numel()

    assert unique_colors <= NUM_CLASSES, "More color classes than label classes."
