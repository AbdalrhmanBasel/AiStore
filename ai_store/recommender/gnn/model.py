# recommender/gnn_model.py

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from recommender.utils import save_encoders

# --- CONFIG ---
CSV_PATH = 'recommender/data/interaction_graph.csv'
EMBEDDINGS_PATH = 'recommender/embeddings/gnn_embeddings.pt'
ENCODERS_PATH = 'recommender/embeddings/encoders.pkl'

# --- Load CSV ---
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded {len(df)} interactions from CSV")

# --- Encode users and products ---
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df['user_id_enc'] = user_encoder.fit_transform(df['user_id'])
df['product_id_enc'] = product_encoder.fit_transform(df['product_id'])

# Save encoders
os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
with open(ENCODERS_PATH, 'wb') as f:
    pickle.dump({
        'user_encoder': user_encoder,
        'product_encoder': product_encoder
    }, f)
print(f"✅ Saved encoders to {ENCODERS_PATH}")

# --- Interaction weights ---
interaction_weights = {
    'view': 1.0,
    'cart': 2.0,
    'wishlist': 2.5,
    'rating': 3.0,
    'purchase': 4.0
}
df['weight'] = df['interaction_type'].map(interaction_weights)

# --- Build edge_index ---
edge_index = torch.tensor([
    df['user_id_enc'].values,
    df['product_id_enc'].values + df['user_id_enc'].max() + 1
], dtype=torch.long)

edge_attr = torch.tensor(df['weight'].values, dtype=torch.float)

# --- Build Graph Data ---
num_users = df['user_id_enc'].nunique()
num_products = df['product_id_enc'].nunique()
num_nodes = num_users + num_products

data = Data(
    edge_index=edge_index,
    edge_attr=edge_attr,
    num_nodes=num_nodes
)

print(f"📊 Graph: {num_users} users, {num_products} products, {len(df)} edges")

# --- GNN Model ---
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --- Train ---
embedding_dim = 64
node_features = nn.Embedding(num_nodes, embedding_dim)
torch.nn.init.xavier_uniform_(node_features.weight)

model = GraphSAGE(embedding_dim, 128, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 11):
    model.train()
    optimizer.zero_grad()

    x = node_features.weight
    out = model(x, data.edge_index, data.edge_attr)

    # Dummy loss: force embeddings to have unit norm
    loss = (out.norm(dim=1).mean() - 1.0).pow(2)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# --- Save embeddings ---
user_emb = out[:num_users]
product_emb = out[num_users:]

os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
torch.save({'user_emb': user_emb, 'product_emb': product_emb}, EMBEDDINGS_PATH)

print(f"✅ Embeddings saved to {EMBEDDINGS_PATH}")
