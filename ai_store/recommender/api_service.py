# recommender/api_service.py

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

EMB_FILE = 'recommender/embeddings/gnn_embeddings.pt'

emb_data = torch.load(EMB_FILE)
user_emb = emb_data['user_emb']
product_emb = emb_data['product_emb']

print(f"✅ Loaded embeddings: {user_emb.shape[0]} users, {product_emb.shape[0]} products")

app = FastAPI(title="Recommendation Service 🚀", version="1.0")

class RecommendRequest(BaseModel):
    user_id_enc: int
    top_n: int = 10

@app.post("/recommend/", response_model=List[int])
def recommend(request: RecommendRequest):
    user_vector = user_emb[request.user_id_enc].unsqueeze(0)

    scores = F.cosine_similarity(user_vector, product_emb)
    top_indices = scores.argsort(descending=True)[:request.top_n]

    return top_indices.tolist()

@app.get("/")
def root():
    return {"message": "Recommendation Service is running 🚀"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005, reload=True)
