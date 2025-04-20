
import pandas as pd
import numpy as np
import re
import ast



def safe_literal_eval(x):
    if isinstance(x, (list, dict)):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return [] if not isinstance(x, dict) else {}
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return x

def parse_price(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    cleaned = re.sub(r'[^0-9.]', '', str(x))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        'average_rating', 'rating_number', 'num_features', 'description_length',
        'price', 'num_images', 'num_videos', 'num_categories', 'num_details'
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df

def clean_meta(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from Amazon metadata DataFrame.
    """
    df = meta_df.copy()

    # 1) Parse JSON‑like columns (lists/dicts stored as strings)
    json_cols = ['categories', 'details', 'features', 'images', 'videos', 'bought_together']
    for col in json_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_literal_eval)
        else:
            # if missing, fill with empty lists/dicts
            default = {} if col == 'details' else []
            df[col] = [default] * len(df)

    # 2) Basic feature counts
    df['num_categories']   = df['categories'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_details']      = df['details'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
    df['num_features']     = df['features'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_images']       = df['images'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_videos']       = df['videos'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # 3) Textual feature
    # df['description_length'] = df.get('description', pd.Series(['']*len(df))).fillna('').apply(lambda s: len(s.split()))
    df['description_length'] = df.get('description', pd.Series(['']*len(df))).fillna('').apply(lambda s: len(str(s).split()))


    # 4) Numeric parsing
    df['average_rating'] = pd.to_numeric(df.get('average_rating', 0), errors='coerce')
    df['rating_number']  = pd.to_numeric(df.get('rating_number', 0), errors='coerce')
    df['price']          = df.get('price', np.nan).apply(parse_price)

    # 5) Handle missing values
    df = handle_missing_values(df)

    return df



# ==== Load and clean data ====

# import os
# from features import create_feature_matrix, feature_engineering, normalize_features

# # --- Define paths ---
# raw_reviews_path = "../../../data/raw/reviews_electronics_small.csv"
# raw_meta_path = "../../../data/raw/metadata_electronics_small.jsonl"
# save_dir = "../../../data/processed"
# os.makedirs(save_dir, exist_ok=True)

# # --- Load raw data ---
# reviews_df, meta_df = load_data(raw_reviews_path, raw_meta_path)

# # --- Clean metadata ---
# cleaned_meta = clean_meta(meta_df)
# cleaned_meta_path = os.path.join(save_dir, "cleaned_meta.csv")
# cleaned_meta.to_csv(cleaned_meta_path, index=False)

# # --- Create and save feature matrix ---
# feature_matrix = create_feature_matrix(cleaned_meta)  # Pass cleaned_meta directly, not a path
# feature_matrix = feature_engineering(feature_matrix)
# feature_matrix = normalize_features(feature_matrix)
# feature_matrix_path = os.path.join(save_dir, "meta_feature_metrix.csv")
# feature_matrix.to_csv(feature_matrix_path, index=False)


# # --- Optional: Save raw metadata sample ---
# # sample_meta_path = os.path.join(save_dir, "sample_meta.jsonl")
# # meta_df.to_json(sample_meta_path, orient="records", lines=True)

# # --- Display outputs ---
# print("Cleaned Metadata Sample:")
# print(cleaned_meta.head(10))
# print("\nFeature Matrix Sample:")
# print(feature_matrix.head(10))
