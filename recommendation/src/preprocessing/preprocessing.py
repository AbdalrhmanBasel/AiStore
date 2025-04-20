# """
# 1. Load sampled reviews
# 2. Extract unique product IDs (ASINs)
# 3. Load matching product metadata
# 4. Clean reviews for graph building
# 5. Save cleaned data to CSV or other formats
# """

# from data_loader import load_reviews_sample, load_meta, clean_reviews
# from graph_builder import build_graph
# import configparser
# import torch
# import os

# config = configparser.ConfigParser()
# config.read('../../config/base_config.ini')

# REVIEWS_DATA_PATH = "../../data/raw/Electronics_5.csv"
# METADATA_DATA_PATH = "../../data/raw/meta_Electronics.jsonl"
# PROCESSED_DATA_PATH = "../../data/processed"
# SAMPLE_SIZE = int(config['SAMPLING']['SAMPLE_SIZE'])
# CHUNK_SIZE = int(config['SAMPLING']['CHUNK_SIZE'])

# print(REVIEWS_DATA_PATH)
# print(SAMPLE_SIZE)

# # os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# reviews_df = load_reviews_sample(REVIEWS_DATA_PATH, sample_size=SAMPLE_SIZE, chunk_size=CHUNK_SIZE)  
# asins = reviews_df['parent_asin'].unique().tolist()

# meta_df = load_meta(METADATA_DATA_PATH, asins_to_keep=asins)
# reviews_df = clean_reviews(reviews_df)

# data = build_graph(reviews_df, meta_df)

# edge_index = data.edge_index
# features = data.x
# labels = None 

# torch.save(edge_index, os.path.join(PROCESSED_DATA_PATH, "edge_index.pt"))
# torch.save(features, os.path.join(PROCESSED_DATA_PATH, "node_features.pt"))
# if labels is not None:
#     torch.save(labels, os.path.join(PROCESSED_DATA_PATH, "labels.pt"))

# print("Data preprocessing completed and saved to the processed directory.")

# edge_index = torch.load(os.path.join(PROCESSED_DATA_PATH, "edge_index.pt"))
# features = torch.load(os.path.join(PROCESSED_DATA_PATH, "node_features.pt"))
# # labels = torch.load(os.path.join(PROCESSED_DATA_PATH, "labels.pt"))

# print(f"Edge Index: {edge_index.shape}")
# print(f"Features: {features.shape}")
# # print(f"Labels: {labels.shape}")


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch

# Load reviews and meta data
reviews_df = pd.read_json('reviews_Electronics_5.json', lines=True)
meta_df = pd.read_json('meta_Electronics.json', lines=True)

# --- USER FEATURES ---
user_stats = reviews_df.groupby("reviewerID").agg({
    "overall": ["mean", "count"]
})
user_stats.columns = ["user_avg_rating", "user_review_count"]
user_stats.reset_index(inplace=True)

# --- ITEM FEATURES ---
item_stats = reviews_df.groupby("asin").agg({
    "overall": ["mean", "count"]
})
item_stats.columns = ["item_avg_rating", "item_review_count"]
item_stats.reset_index(inplace=True)

# Join with meta data for price and category
meta_df = meta_df[["asin", "price", "category"]]
item_features = pd.merge(item_stats, meta_df, on="asin", how="left")

# Clean up and fill missing values
item_features["price"] = item_features["price"].fillna(item_features["price"].mean())
item_features["category"] = item_features["category"].fillna("Unknown")

# Normalize price and review count
scaler = MinMaxScaler()
item_features[["item_review_count", "price"]] = scaler.fit_transform(item_features[["item_review_count", "price"]])
user_stats[["user_review_count"]] = scaler.fit_transform(user_stats[["user_review_count"]])

# One-hot encode category
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
cat_encoded = encoder.fit_transform(item_features[["category"]])
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(["category"]))
item_features = pd.concat([item_features.drop("category", axis=1), cat_encoded_df], axis=1)

# --- CREATE NODE FEATURE MATRIX ---

# Create a mapping of user_id and item_id to node index
user_id_map = {user_id: idx for idx, user_id in enumerate(user_stats["reviewerID"])}
item_id_map = {asin: idx for idx, asin in enumerate(item_features["asin"])}

# Total number of nodes = users + items
num_users = len(user_stats)
num_items = len(item_features)
num_nodes = num_users + num_items

# Feature dimension: depends on how many features you add
feature_dim = user_stats.shape[1] - 1 + item_features.shape[1] - 1

# Create empty feature matrix
node_features = np.zeros((num_nodes, feature_dim), dtype=np.float32)

# Fill in user features
for i, row in user_stats.iterrows():
    node_id = user_id_map[row["reviewerID"]]
    node_features[node_id, :2] = [row["user_avg_rating"], row["user_review_count"]]

# Fill in item features
for i, row in item_features.iterrows():
    node_id = item_id_map[row["asin"]] + num_users
    features = row.drop("asin").values.astype(np.float32)
    node_features[node_id] = features

# Save the node features as .pt file
torch.save(torch.tensor(node_features), "data/node_features.pt")

print(f"Node features saved to data/node_features.pt")
