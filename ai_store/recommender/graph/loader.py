# recommender/graph/loader.py

import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import os
from recommender.utils import save_encoders

INPUT_CSV = 'recommender/data/interaction_graph.csv'
OUTPUT_PT = 'recommender/data/interaction_graph.pt'

def build_graph_data():
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Loaded {len(df)} interactions")

    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    df['user_id_enc'] = user_encoder.fit_transform(df['user_id'])
    df['product_id_enc'] = product_encoder.fit_transform(df['product_id'])

    save_encoders(user_encoder, product_encoder)

    interaction_weight = {
        'view': 1.0,
        'cart': 2.0,
        'wishlist': 2.5,
        'purchase': 3.0,
    }

    df['weight'] = df['interaction_type'].map(interaction_weight)

    edge_index = torch.tensor([
        df['user_id_enc'].values,
        df['product_id_enc'].values + df['user_id_enc'].max() + 1
    ], dtype=torch.long)

    edge_attr = torch.tensor(df['weight'].values, dtype=torch.float)

    num_users = df['user_id_enc'].nunique()
    num_products = df['product_id_enc'].nunique()
    num_nodes = num_users + num_products

    print(f"📊 {num_users} users, {num_products} products, {len(df)} edges")

    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )

    os.makedirs(os.path.dirname(OUTPUT_PT), exist_ok=True)
    torch.save(data, OUTPUT_PT)
    print(f"✅ Saved graph to {OUTPUT_PT}")
