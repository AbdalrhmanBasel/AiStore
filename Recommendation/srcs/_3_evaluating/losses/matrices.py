from torchmetrics.retrieval import RetrievalRecall, RetrievalPrecision, RetrievalNormalizedDCG

def calculate_metrics(scores, labels, k=10):
    print(f"Scores shape: {scores.shape}")
    print(f"Labels shape: {labels.shape}")
    
    precision = RetrievalPrecision(top_k=k)(scores, labels)
    recall = RetrievalRecall(top_k=k)(scores, labels)
    ndcg = RetrievalNormalizedDCG(top_k=k)(scores, labels)
    
    return {'precision@k': precision, 'recall@k': recall, 'ndcg@k': ndcg}
