## 🧠 **Big Picture: What’s our end goal?**

Build a **GNN-based recommendation system** that can:

1. **Learn** meaningful node embeddings.
2. **Predict/Recommend** product-user relevance.
3. **Scale** to large datasets using mini-batch training.

---

### ✅ **Workflow in Order (with your question in mind):**

#### 1. **Preprocess Raw Data**
> *This is your "construct the graph" step.*

- Convert CSVs or JSONs into a graph:
  - Nodes: users, products
  - Edges: interactions like purchases or ratings
  
- Construct a PyG `Data` object with:
  - `x`: node features
  - `edge_index`: edge list
  - `y`: labels (if any)
  - `train_mask`, `val_mask`, `test_mask`: node splits

🔹 **Output:** A graph object saved as `processed_graph.pt`

---

#### 2. **Load the Preprocessed Graph**

> *This is where you "load the big data".*

- Load `processed_graph.pt` with `torch.load`
- It contains the full graph in memory

```python
data = torch.load("artifacts/processed_graph.pt")
```

🔹 **Now you have your entire graph ready to feed into training.**

---

#### 3. **Initialize Graph DataLoaders**

> *We don’t feed the full graph to the model every time — we use a DataLoader to sample small subgraphs.*

- We use **`NeighborLoader`** to sample subgraphs:
  - Trains the model on mini-batches
  - Each batch = a node + sampled neighbors
- DataLoaders are split into:
  - `train_loader`
  - `val_loader`
  - `test_loader`

```python
train_loader, val_loader, test_loader = get_data_loaders(data)
```

---

#### 4. **Train the GNN Model (GraphSAGE)**

> *This is the learning phase.*

- You loop over batches from `train_loader`
- Each batch is a mini-graph
- Forward pass through GraphSAGE
- Compute loss using `batch.y`
- Backpropagate & update weights

---

#### 5. **Evaluate & Recommend**

- Use `val_loader` and `test_loader` to evaluate the model
- Extract embeddings or predictions
- Generate recommendations:
  - top-5 products for a user
  - Based on cosine similarity or score

---

### 🔄 To summarize your mental model:

```txt
Raw CSV/JSON 
   ↓
Preprocessing → PyG Data object
   ↓
Load data → Full graph in memory
   ↓
NeighborLoader → Sample mini-batches
   ↓
GraphSAGE training → Learn embeddings
   ↓
Evaluation & Recommendation
```

---

### 🧭 Next Steps

You're now ready to move into **GraphSAGE implementation**. Would you like to:
1. Build the GraphSAGE model class?
2. Build the training loop for it?
3. Extract embeddings & visualize?

Let’s pick the next milestone together.

---
Absolutely — here's your Harvard-level structured **To-Do List** for building the GraphSAGE-based recommendation system step-by-step:

---

### ✅ **Phase 1: Preprocessing & Graph Construction**

1. [x] Load raw data (reviews, metadata) from Amazon Electronics dataset.
2. [x] Construct nodes and edges:
   - [x] Nodes: users and products
   - [x] Edges: interactions (ratings, purchases)
3. [x] Extract node features (categories, embeddings, counts)
4. [x] Build `torch_geometric.data.Data` object:
   -[ ] `x`, `edge_index`, `y` (if classification)
5. [x] Split nodes into:
   - [x] `train_mask`
   - [x] `val_mask`
   - [x] `test_mask`
6. [x] Save graph object as `processed_graph.pt`

---

### ✅ **Phase 2: Data Loading (Mini-batching)**

7. [ ] Load `processed_graph.pt`
8. [ ] Create `NeighborLoader` for:
   - [ ] Training loader
   - [ ] Validation loader
   - [ ] Test loader
9. [ ] Set sampling parameters: `batch_size`, `num_neighbors`, `shuffle`, etc.

---

### ✅ **Phase 3: Model Design & Training**

10. [ ] Define `GraphSAGE` model class:
    - [ ] Layers: `SAGEConv`, activation, dropout
    - [ ] Forward pass logic
11. [ ] Create training loop:
    - [ ] Iterate over `train_loader`
    - [ ] Compute loss (`CrossEntropyLoss`)
    - [ ] Backpropagation and optimizer step
12. [ ] Implement evaluation logic using `val_loader`
13. [ ] Save trained model checkpoint

---

### ✅ **Phase 4: Embedding Extraction & Recommendation**

14. [ ] Extract node embeddings using the trained model
15. [ ] Compute similarity scores between users and products
16. [ ] Generate top-N recommendations per user
17. [ ] (Optional) Visualize node embeddings using TSNE or UMAP

---

### ✅ **Phase 5: Reporting & Utilities**

18. [ ] Implement `print_graph_report(graph_dataset)`
19. [ ] Implement `get_graph_report(graph_dataset, render_as='json' or 'dict')`
20. [ ] Add CLI or script entry point for:
    - [ ] Preprocessing
    - [ ] Training
    - [ ] Evaluation
    - [ ] Recommending

---

Would you like me to generate this as a Markdown checklist you can paste into your `README.md` or `project_notes.md` file in Obsidian?


Certainly! Let’s dive deeper into each phase of the project for building a GNN-powered recommendation system tailored to eCommerce platforms, covering all steps, methodologies, and best practices.

---

# 🧠 Building a GNN-Powered Recommendation System for eCommerce

In eCommerce, personalized recommendations are essential for enhancing user engagement and driving sales. Traditional recommendation algorithms often struggle to effectively capture the complex, multi-relational interactions between users and items. Graph Neural Networks (GNNs) can overcome these challenges by capturing such intricate relationships in a way that traditional methods cannot. This article details the process of developing a GNN-based recommendation system for an eCommerce platform, leveraging state-of-the-art methods for preprocessing, model training, evaluation, and deployment.

---

## 📌 Project Phases Overview

The development of a GNN-powered recommendation system involves multiple stages. These stages ensure the solution is modular, scalable, and production-ready, which is necessary for handling large-scale eCommerce environments:

1. **Data & Graph Construction**
2. **Model Training (GraphSAGE)**
3. **Recommendation Inference**
4. **Django Integration**
5. **Evaluation & Dashboard**
6. **Hyperparameter Tuning**
7. **Model Optimization & Deployment**
8. **Scalability & Batch Processing**
9. **Post-Launch Monitoring & Feedback Loop**

---

## 📋 Phase 1: Data & Graph Construction

### 1.1 Data Acquisition

For this project, we will use the **Amazon Electronics 5-core dataset**. This dataset contains user-item interaction data, including metadata, reviews, and item links. Here’s how to proceed:

- **Data Download**: The dataset can be obtained from [Amazon Reviews Dataset](https://amazon-reviews-2023.github.io/data_processing/5core.html). It contains user ratings for products, along with metadata such as product categories, titles, and brands.

- **Data Preprocessing**:
  - **Text Cleaning**: Review texts may contain unnecessary characters, stop words, and other noise. Preprocess the text using common NLP techniques such as:
    - Tokenization
    - Removing stop words
    - Lemmatization
    - Lowercasing
  - **Feature Engineering**: 
    - **Item Features**: Encode categorical features (e.g., product categories) using one-hot encoding. For textual features like product titles, employ TF-IDF vectorization or advanced models like BERT embeddings for better semantic understanding.
    - **User Features**: Calculate aggregate statistics such as the number of interactions per user, average ratings, and product category preferences.

### 1.2 Data Graph Construction

- **Mapping Users and Items**: Each user and item in the dataset must be uniquely identified. Map them to integer IDs for graph construction:
  - Create a `user2id` mapping and an `item2id` mapping.

- **Graph Construction**:
  - The core idea is to represent the data as a **bipartite graph** where users and items are two types of nodes connected by edges that represent interactions (e.g., a rating or purchase).
  - **Edges**: Each interaction between a user and an item is represented by an edge. If a user has rated or interacted with a product, an edge is created between the user and the item.
  - **Edge Index**: Use `edge_index`, a tensor representing the source and target nodes for each edge, which is a crucial structure in graph-based deep learning frameworks like PyTorch Geometric.

- **Feature Matrix**:
  - **Item Features**: For each item, create a feature vector that could contain attributes like one-hot encoded categories, normalized price, and embedded product titles.
  - **User Features**: For each user, create a feature vector that includes statistics like the number of interactions, average rating, and user-specific preferences.

- **Save the Preprocessed Data**:
  - Store the graph in an efficient format like `.pt` (PyTorch tensor format) or `.npy` (NumPy format).
  - Example files:
    - `edge_index.pt`: The graph edges.
    - `features.pt`: The node features for users and items.
    - `labels.pt`: Target labels (for example, implicit feedback of interactions).
    - `mappings.pkl`: The mappings of user and item IDs.

---

## 🧠 Phase 2: Model Training (GraphSAGE)

### 2.1 Implement GraphSAGE Model

- **Why GraphSAGE?** GraphSAGE (Graph Sample and Aggregation) is a powerful technique for learning node embeddings by sampling a fixed-size neighborhood of each node and aggregating information from neighboring nodes. This is particularly useful for large graphs with millions of nodes.

- **GraphSAGE Model Implementation**:
  - **Layer-wise Aggregation**: The GraphSAGE model works by aggregating the features of neighboring nodes in multiple layers. The key aggregation techniques include:
    - Mean aggregation
    - LSTM-based aggregation
    - Pooling-based aggregation
  - Implement a GraphSAGE model using **PyTorch Geometric**. The basic architecture should consist of the following:
    - **Embedding Layers**: Initial embeddings for users and items.
    - **GraphSAGE Layer**: A layer that aggregates information from neighbors, transforming the node representations layer by layer.
    - **Fully Connected Layer**: A fully connected layer for prediction based on the aggregated node embeddings.

- **Model Code**:
  ```python
  class GraphSAGE(torch.nn.Module):
      def __init__(self, in_channels, out_channels, num_layers):
          super(GraphSAGE, self).__init__()
          self.convs = torch.nn.ModuleList()
          self.convs.append(SAGEConv(in_channels, out_channels))
          for _ in range(num_layers - 1):
              self.convs.append(SAGEConv(out_channels, out_channels))

      def forward(self, x, edge_index):
          for conv in self.convs:
              x = conv(x, edge_index)
          return x
  ```

### 2.2 Training the Model

- **Data Loading**:
  - Use **NeighborLoader** from PyTorch Geometric to load mini-batches of data. It ensures that only the necessary parts of the graph are loaded into memory for each batch, making training efficient even for large graphs.
  - **DataLoader Example**:
    ```python
    from torch_geometric.loader import NeighborLoader
    train_loader = NeighborLoader(
        data, 
        num_neighbors=[10, 10],  # Number of neighbors sampled at each layer
        batch_size=64, 
        shuffle=True
    )
    ```

- **Loss Function**:
  - For implicit feedback (i.e., binary interactions), use **Binary Cross-Entropy** loss with negative sampling:
    ```python
    criterion = torch.nn.BCEWithLogitsLoss()
    ```

- **Training Loop**:
  - Use an optimizer like **Adam** or **AdamW**.
  - Implement early stopping based on validation performance to avoid overfitting.

- **Model Saving**:
  - After training, save the model weights and embeddings:
    ```python
    torch.save(model.state_dict(), 'model.pth')
    ```

---

## ⚡ Phase 3: Recommendation Inference

### 3.1 Generating Recommendations

- **Inference Flow**:
  - After training, we need to generate personalized recommendations for users. The process involves:
    - Loading the trained model.
    - Retrieving the embeddings for all users and items.
    - Computing the similarity score (dot product or MLP-based) between each user's embedding and all items' embeddings.

- **Recommendation Code**:
  ```python
  def recommend(user_id, top_k=10):
      user_embedding = model(user_features[user_id])
      item_embeddings = model(item_features)
      scores = torch.matmul(user_embedding, item_embeddings.T)
      top_items = torch.topk(scores, top_k)
      return top_items
  ```

- **Handling Cold-Start Problem**:
  - Implement solutions for the cold-start problem, where new users or items don’t have interaction history. You could use **content-based filtering** as a fallback approach for cold-start scenarios.

---

## 🌐 Phase 4: Django Integration

### 4.1 Set Up Django

- **Create Django App**: In your Django project, create an app (`recengine`) dedicated to handling recommendations. Install necessary dependencies such as **Django REST Framework**.

- **Recommendation API**:
  - Create an API endpoint `/recommendations/<user_id>/` that returns recommendations for the given user. The logic here will interface with the trained model to fetch top-K items.
  
- **API Example**:
  ```python
  from rest_framework.views import APIView
  from rest_framework.response import Response

  class RecommendationView(APIView):
      def get(self, request, user_id):
          recommendations = recommend(user_id)
          return Response({"recommendations": recommendations.tolist()})
  ```

- **Frontend Integration**:
  - In the frontend, add a "Recommended for You" section that queries this API to display the personalized recommendations for users.

---

## 📊 Phase 5: Evaluation & Dashboard

### 5.1 Performance Evaluation

- **Evaluation Metrics**: Use various metrics to evaluate the recommendation system's quality:
  - **Precision@K**: Measures the fraction of relevant items in the top-K recommendations.
  - **Recall@K**: Measures the fraction of relevant items that are successfully retrieved in the top-K.
  - **NDCG@K**: Normalized Discounted Cumulative Gain evaluates ranked lists and accounts for the position of items.

- **Cold-Start Evaluation**: Measure how the model performs when new users/items are introduced, possibly leveraging additional features such as item metadata or user demographics.

- **Visualization**: Use libraries like **Matplotlib** or **Seaborn** to visualize the results of the evaluation metrics (e.g., precision-recall curves).

---

## 🛠️ Phase 6: Hyperparameter Tuning

### 6.1 Tuning with Optuna

- **Hyperparameters to Tune**: Focus on learning rate, number of layers, hidden dimension sizes, dropout rate, and neighbor sampling settings.
- **Use Optuna** for an automated hyperparameter search:
  ```python
  import optuna
  def objective(trial):
      lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
      hidden_dim = trial.suggest_int('hidden_dim', 16, 256)
      dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
      # Setup model, loss, optimizer here and return validation loss
  study = optuna.create_study(direction="minimize")
  study.optimize(objective, n_trials=100)
  ```

---

## 🏆 Phase 7: Model Optimization & Deployment

### 7.1 Optimizing the Model

- **TorchScript**: Convert the trained model to **TorchScript** for optimized inference:
  ```python
  scripted_model = torch.jit.script(model)
  scripted_model.save('model.pt')
  ```

- **Containerization**: Use **Docker** to create a container for both the model and the Django app, ensuring consistency across environments.

- **Deployment on Cloud**: Deploy the model on a cloud platform such as AWS, GCP, or Azure to handle scaling and load balancing. You can use tools like **AWS Sagemaker** or **Google AI Platform** to streamline deployment.

- **CI/CD Pipeline**: Implement a CI/CD pipeline to automate testing, deployment, and scaling of your recommendation system.

---

## 🔄 Phase 8: Scalability & Batch Processing

### 8.1 Handling Scale

As your user base grows, scaling the recommendation system becomes critical. Use techniques like:
- **Batch Processing**: Process user recommendations in batches, especially during inference.
- **Model Sharding**: Split large models into smaller parts for distributed training or inference.

---

## 📊 Phase 9: Post-Launch Monitoring & Feedback Loop

### 9.1 Post-Launch Monitoring

Once deployed, it's essential to continuously monitor the model’s performance:
- Track **click-through rates (CTR)**, **conversion rates**, and **user engagement** metrics.
- Collect user feedback on recommendations and retrain the model periodically to improve its predictions.

### 9.2 Feedback Loop

- Implement a **feedback loop** where the model is retrained regularly with new user interactions and ratings.
- Ensure that the system evolves based on changing user preferences.

---

This detailed approach provides a comprehensive framework to develop and deploy a GNN-powered recommendation system for an eCommerce platform, from data preprocessing to evaluation and post-launch monitoring.