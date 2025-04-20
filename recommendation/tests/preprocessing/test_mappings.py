import pandas as pd
import tempfile
from src.preprocessing.mappings import create_id_mappings, apply_mappings, save_mappings

def test_create_id_mappings_returns_correct_length():
    df = pd.DataFrame({
        'reviewerID': ['U1', 'U2', 'U1', 'U3'],
        'asin': ['P1', 'P2', 'P1', 'P3']
    })

    reviewer_to_id, asin_to_id = create_id_mappings(df)
    
    assert len(reviewer_to_id) == 3, f"Expected 3 unique reviewer IDs, got {len(reviewer_to_id)}"
    assert len(asin_to_id) == 3, f"Expected 3 unique product IDs, got {len(asin_to_id)}"

def test_apply_mappings_with_unseen_ids():
    df = pd.DataFrame({
        'reviewerID': ['U1', 'U2'],
        'asin': ['P1', 'P2']
    })
    
    reviewer_to_id, asin_to_id = create_id_mappings(df)
    
    unseen_df = pd.DataFrame({
        'reviewerID': ['U3'],
        'asin': ['P1']
    })
    result = apply_mappings(unseen_df, reviewer_to_id, asin_to_id)
    
    assert result['reviewerID'][0] == 2, "Expected ID 2 for unseen reviewer 'U3'"
    assert result['asin'][0] == 0, "Expected ID 0 for seen product 'P1'"

def test_save_and_load_mappings():
    df = pd.DataFrame({
        'reviewerID': ['U1', 'U2'],
        'asin': ['P1', 'P2']
    })

    reviewer_to_id, asin_to_id = create_id_mappings(df)

    # Create a temporary file to save mappings
    with tempfile.TemporaryDirectory() as temp_dir:
        save_mappings(reviewer_to_id, asin_to_id, temp_dir)
        loaded_reviewer_to_id, loaded_asin_to_id = load_mappings(temp_dir)

        assert reviewer_to_id == loaded_reviewer_to_id, "Loaded reviewer_to_id does not match"
        assert asin_to_id == loaded_asin_to_id, "Loaded asin_to_id does not match"
