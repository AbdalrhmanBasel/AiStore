# from src.models.graphsage import GraphSAGEModel
# from src.utils.load_data import load_data
# from train import train
# from torch.optim import Adam
# import torch

# # Load the data
# data = load_data()
# train_pos, train_neg = data.train_pos, data.train_neg

# # Initialize the GraphSAGE model
# in_channels = data.x.shape[1]  # Number of input features per node (e.g., if each node has 10 features, this is 10)
# hidden_channels = 128  # Number of hidden channels in the GraphSAGE layers
# out_channels = 64  # Output embedding size
# model = GraphSAGEModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)

# # Set up the optimizer (Adam)
# optimizer = Adam(model.parameters(), lr=0.001)

# # Train the model
# train(model, data, optimizer, train_pos, train_neg, epochs=100)
# def train_mode():
#     # Load processed data
#     edge_index = torch.load("./data/processed/edge_index.pt")
#     features = torch.load("./data/processed/features.pt")

#     # Prepare dataset
#     dataset = LinkPredictionDataset(edge_index, num_nodes=features.size(0))
#     data = dataset.get_data(features)
#     train_pos, train_neg = dataset.train_edges, dataset.train_neg_edges
#     test_pos, test_neg = dataset.test_edges, dataset.test_neg_edges

#     # Initialize model and optimizer
#     model = GraphSAGEModel(in_channels=features.size(1), hidden_channels=64, out_channels=32)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     # Training loop
#     for epoch in range(1, 1001):
#         loss = train(model, data, optimizer, train_pos, train_neg)
#         if epoch % 10 == 0:
#             acc, auc, precision, recall, f1, _ = eval_metrics(model, data, test_pos, test_neg)
            
#             model.eval()
#             with torch.no_grad():
#                 z = model(data.x, data.edge_index)
#                 mrr, hits = evaluate_link_prediction(z, test_pos, test_neg, k=10)

#             print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f} | "
#                   f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
#                   f"MRR: {mrr:.4f} | Hits@10: {hits:.4f}")

#     # After training, compute final embeddings
#     model.eval()
#     with torch.no_grad():
#         z = model(data.x, data.edge_index)

#     # Visualize embeddings
#     # visualize_embeddings(z)

#     # Save model and embeddings
#     save_model_outputs(model, z)