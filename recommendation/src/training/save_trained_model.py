import torch

def save_trained_model(model, embeddings, model_path, embed_path):
    torch.save(model.state_dict(), model_path)
    torch.save(embeddings, embed_path)
    print("✅ Model and embeddings saved.")
