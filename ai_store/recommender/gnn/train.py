# recommender/gnn/train.py

import torch
import torch.nn as nn
from recommender.graph.loader import build_graph_data, OUTPUT_PT
from recommender.gnn.model import GraphSAGE

def train_gnn():
    build_graph_data()
    data = torch.load(OUTPUT_PT, weights_only=False)

    embedding_dim = 64
    node_features = nn.Embedding(data.num_nodes, embedding_dim)
    torch.nn.init.xavier_uniform_(node_features.weight)

    model = GraphSAGE(in_channels=embedding_dim, hidden_channels=128, out_channels=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 11):
        model.train()
        optimizer.zero_grad()

        x = node_features.weight
        out = model(x, data.edge_index, data.edge_attr)

        loss = (out.norm(dim=1).mean() - 1.0).pow(2)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    user_emb = out[:data.edge_index[0].max() + 1]
    product_emb = out[data.edge_index[0].max() + 1:]

    torch.save({'user_emb': user_emb, 'product_emb': product_emb}, 'recommender/embeddings/gnn_embeddings.pt')
    print("✅ Saved embeddings: recommender/embeddings/gnn_embeddings.pt")
