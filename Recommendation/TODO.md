# 📊 GNN Recommendation System Development

## Overview

This recommendation system is a state-of-the-art Graph Neural Network (GNN) recommendation system designed for eCommerce platforms. This system leverages graph-based techniques to provide personalized recommendations to used based on their interactions with items (products, reviews, and metadata). 

The project is designed not only to dominstrate that GNN recommendation systems are very reliable recommenderation systems for eCommerce. Additionally, the project aims to showcase best practices for building scalabe, moduler, and production-ready recommendation systems using PyTorch and GNNs.

##  Project Phases: Overview

1. **Phase #1:** Data & Graph Construction.
2. **Phase #2:** Model Training (GraphSAGE).
3. **Phase #3:** Recommendation Inference.
4. **Phase #4:** Django Integration.
5. **Phase #5:** Evaluation & Dashboard. 
6. **Phase #6:** Hypterparameter Tuning.
7. **Phase #7:** Model Optimization & Deployment.
8. **Phase #8:** Svalability & Batch Processing.
9. **Phase #9:** Post-Launch Monitorying & Feedback Loop.


## 📋 Checklist

### 🧑‍💻 Phase #1: Data & Graph Construction
- [x] Download `meta`, `reviews`, and `links` from the [Amazon Electronics (5-core dataset)](https://amazon-reviews-2023.github.io/data_processing/5core.html)
- [x] Clean and preprocess review texts (optional NLP)
- [x] Create `user2id` and `item2id` mappings for users and items
- [x] Map all datasets to integer IDs (users, items, etc.)
- [x] Build bipartite graph (`user ↔ item`) based on user-item interactions
- [ ] Extract and preprocess item features:
  - [ ] One-hot encode `category`
  - [ ] Embed `title` using TF-IDF or BERT
  - [ ] Normalize `price` and encode `brand`
- [ ] Extract user features:
  - [ ] Calculate user history size (number of interactions)
  - [ ] Calculate average rating for each user
- [x] Save processed data:
  - [x] `edge_index.pt`: Graph structure (user-item edges)
  - [x] `features.pt`: Node features (item/user features)
  - [x] `labels.pt`: Target labels (interactions, ratings)
  - [x] `mappings.pkl`: Mappings for user and item IDs

## 🧠 PHASE 2: Model Training (GraphSAGE)

- [x] Implement GraphSAGE model in `gnn_model.py`
- [x] Use `NeighborLoader` from PyTorch Geometric for mini-batch training
- [ ] Define scoring head (dot product or MLP)
- [x] Define loss function (binary cross-entropy with negative sampling)
- [x] Add evaluation metrics:
  - [x] Precision@K
  - [x] Recall@K
  - [x] NDCG@K (Normalized Discounted Cumulative Gain)
- [ ] Perform data splitting (80% train, 10% validation, 10% test)
- [ ] Implement early stopping during training based on validation performance
- [ ] Save trained model:
  - [ ] Model weights (`model.pth`)
  - [ ] User and item embeddings for inference
- [ ] Log training results (loss, accuracy, metrics)

## ⚡ PHASE 3: Recommendation Inference

- [ ] Implement inference logic in `recommender.py`
- [ ] Load trained model and embeddings
- [ ] For each user:
  - [ ] Retrieve the user’s embedding vector
  - [ ] Compute the dot product with all item embeddings
  - [ ] Mask items that the user has already interacted with
  - [ ] Return top-K recommended items (Top-K ASINs)
- [ ] Save recommendations to a `.json` file or database for use in the web application

## 🌐 PHASE 4: Django Integration

- [ ] Set up Django project with `recengine/` app for recommendation engine
- [ ] Implement recommendation API:
  - [ ] Endpoint: `/recommend/<user_id>/`
  - [ ] Pass `user_id` and return a list of recommended items
- [ ] Integrate model inference with the API endpoint
- [ ] Serialize recommendations into JSON format for frontend consumption
- [ ] Implement frontend “Recommended for You” section:
  - [ ] Display Top-K items for the logged-in user
- [ ] (Optional) Use **Celery** for asynchronous batch updates:
  - [ ] Update recommendations for users periodically in the background

## 📊 PHASE 5: Evaluation & Dashboard

- [ ] Create an evaluation notebook (`eval_notebook.ipynb`)
- [ ] Measure key evaluation metrics:
  - [ ] Precision@5, Precision@10
  - [ ] Recall@5, Recall@10
  - [ ] NDCG@10
  - [ ] Cold-start performance (evaluate on new users/items)
- [ ] Compare recommendation performance against:
  - [ ] Random recommendations
  - [ ] Most Popular items
- [ ] (Optional) Build a dashboard to visualize performance and metrics:
  - [ ] Use **Streamlit** or **Dash** for an interactive dashboard
  - [ ] Display metrics, loss curves, and recommendation examples


## 🛠️ PHASE 6: Code Refactoring & Modularity

- [ ] Refactor project code for better readability and modularity:
  - [ ] Split code into logical modules (data preprocessing, model training, inference, etc.)
  - [ ] Use clear class and function definitions with proper docstrings
  - [ ] Follow best practices for project structure and organization
- [ ] Create configuration files for easy setup (`config.yaml`)
- [ ] Add logging and debugging information for easy tracking of the training and inference process


## 🔄 PHASE 7: Hyperparameter Tuning

- [ ] Perform hyperparameter tuning for the GraphSAGE model:
  - [ ] Learning rate, hidden layers, and embedding dimensions
  - [ ] Dropout rate and neighbor sampling settings
- [ ] Use tools like **Optuna** or **Ray Tune** to automate the hyperparameter search
- [ ] Evaluate the performance of the tuned model against the baseline

## 🏆 PHASE 8: Model Optimization & Deployment

- [ ] Optimize the model for production deployment:
  - [ ] Convert model to TorchScript for optimized inference
  - [ ] Compress the model to reduce memory and inference time
  - [ ] Ensure that the model can handle real-time recommendation requests efficiently
- [ ] Prepare model for deployment:
  - [ ] Containerize the model and API using **Docker**
  - [ ] Set up a CI/CD pipeline for automatic deployment (using GitHub Actions)
  - [ ] Deploy the model to a cloud environment (AWS, Google Cloud, or Azure)
  

## ⚙️ PHASE 9: Scalability & Batch Processing
- [ ] Implement batch processing for large-scale inference:
  - [ ] Use **Dask** or **Apache Spark** for large-scale data processing
  - [ ] Store recommendations in a distributed database (**Cassandra** or **Amazon DynamoDB**)
- [ ] Implement load balancing for handling high traffic:
  - [ ] Use **Kubernetes** for automatic scaling
  - [ ] Optimize the model's ability to serve multiple users concurrently


## 🌟 PHASE 10: Post-Launch Monitoring & Feedback Loop
- [ ] Set up monitoring for real-time performance of the recommendation system:
  - [ ] Track recommendation accuracy and latency
  - [ ] Monitor server and API health using tools like **Prometheus** and **Grafana**
- [ ] Collect feedback from users on recommendation quality:
  - [ ] Use A/B testing to evaluate different models or settings
  - [ ] Gather user ratings on recommended items
- [ ] Continuously improve the system based on feedback:
  - [ ] Retrain models periodically with new data
  - [ ] Implement cold-start solutions for new users or items
