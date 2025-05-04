import torch
from srcs.models.GraphSAGEModel import GraphSAGEModel
from srcs.utils.settings import DEVICE

def load_trained_model(model_path):
    model = GraphSAGEModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model
