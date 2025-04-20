"""
Test Suite for preprocessing.py
===============================
Validates the data preprocessing pipeline for StoreX recommendation system:
- Sampling Amazon dataset
- Cleaning operations
- Graph construction
- Tensor saving and loading
"""

import os
import torch
import pandas as pd
import pytest
from pathlib import Path
from src.preprocessing.preprocessing import (
    sample_amazon_data,
    clean_metadata,
    build_item_item_graph,
    save_graph_data
)


@pytest.fixture
def raw_data(tmp_path):
    """Simulate a small Amazon-style dataset"""
    reviews = pd.DataFrame({
        'reviewerID': ['U1', 'U2', 'U1', 'U3'],
        'asin': ['A1', 'A2', 'A2', 'A3'],
        'overall': [5, 4, 5, 3]
    })
    metadata = pd.DataFrame({
        'asin': ['A1', 'A2', 'A3'],
        'title': ['Product 1', 'Product 2', 'Product 3'],
        'price': [10.0, 15.5, 8.0]
    })
    return reviews, metadata


def test_sample_amazon_data_returns_correct_size(raw_data):
    reviews, metadata = raw_data
    sampled_reviews, sampled_metadata = sample_amazon_data(reviews, metadata, sample_size=2)
    assert len(sampled_reviews) <= 2, "Sampled reviews exceed expected sample size"
    assert all(sampled_reviews['asin'].isin(sampled_metadata['asin'])), \
        "Sampled reviews contain products not in metadata"


def test_clean_metadata_removes_nan_titles():
    metadata = pd.DataFrame({
        'asin': ['A1', 'A2', 'A3'],
        'title': ['Product 1', None, 'Product 3']
    })
    cleaned = clean_metadata(metadata)
    assert cleaned.shape[0] == 2, "Rows with missing titles should be removed"
    assert 'A2' not in cleaned['asin'].values, "Product with missing title not removed"



def test_build_item_item_graph_returns_valid_tensors(raw_data):
    reviews, _ = raw_data
    edge_index, features, labels = build_item_item_graph(reviews)

    assert edge_index.shape[0] == 2, "Edge index should be shape (2, E)"
    assert features.dim() == 2, "Features must be 2D tensor"
    assert labels.dim() == 1, "Labels must be 1D tensor"
    assert features.shape[0] == labels.shape[0], "Each node should have a label and features"


def test_graph_node_ids_within_bounds(raw_data):
    reviews, _ = raw_data
    edge_index, features, _ = build_item_item_graph(reviews)
    num_nodes = features.shape[0]
    assert torch.all(edge_index < num_nodes), "Edge index has invalid node references"


def test_save_graph_data_creates_files(tmp_path, raw_data):
    reviews, _ = raw_data
    edge_index, features, labels = build_item_item_graph(reviews)

    save_graph_data(tmp_path, edge_index, features, labels)

    assert (tmp_path / 'edge_index.pt').exists(), "edge_index.pt not saved"
    assert (tmp_path / 'features.pt').exists(), "features.pt not saved"
    assert (tmp_path / 'labels.pt').exists(), "labels.pt not saved"

    loaded_edge_index = torch.load(tmp_path / 'edge_index.pt')
    assert torch.equal(edge_index, loaded_edge_index), "Saved edge_index is incorrect"


def test_graph_with_no_reviews_raises_error():
    reviews = pd.DataFrame(columns=['reviewerID', 'asin', 'overall'])
    with pytest.raises(ValueError):
        _ = build_item_item_graph(reviews)


def test_handling_duplicate_reviews():
    reviews = pd.DataFrame({
        'reviewerID': ['U1', 'U1'],
        'asin': ['A1', 'A1'],
        'overall': [5, 5]
    })
    edge_index, features, labels = build_item_item_graph(reviews)
    assert edge_index.shape[1] <= 2, "Duplicate reviews should not create extra edges"
