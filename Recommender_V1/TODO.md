# StoreXGNN Development Reference: Task Documentation (Detailed with Checklist)

This document outlines the comprehensive implementation plan for StoreXGNN — a Graph Neural Network-based recommendation system to be integrated into a Django eCommerce platform. Each task is described in detail, specifying what to do, how to do it, and what deliverables must be produced.

---

## PHASE 1: Project Setup and Boilerplate

### ☐ Task 1: Project Initialization

**What to Do:**
Set up the foundational structure and tools for the project to ensure modularity, maintainability, and clarity.
**Steps:**

1. Create the following folder structure:

   * `storex_gnn/`

     * `src/` – source code modules
     * `data/` – raw and processed datasets
     * `models/` – model weights and checkpoints
     * `embeddings/` – generated node embeddings
     * `api/` – Django REST API integration
     * `test/` – unit and integration tests
2. Set up virtual environment and install dependencies.
3. Create a `requirements.txt` file to document dependencies.
4. Add `config.py` to centralize all settings: file paths, hyperparameters, logging, etc.
5. Add a main launcher script `main.py` to orchestrate the workflow.
   **Deliverables:**

* Clean directory structure
* `requirements.txt`
* `config.py`
* `main.py`

---

## PHASE 2: GNN Training Pipeline

### ☐ Task 2: GraphBuilder Class (`preprocessing.py`)

**What to Do:**
Build the graph from the dataset and prepare inputs for the GNN model.
**Steps:**

1. Load Amazon metadata and reviews.
2. Clean data: remove duplicates, missing values, and filter inactive users/items.
3. Create unique integer mappings for user and item IDs.
4. Build a bipartite graph with `edge_index` (user → item).
5. Encode features (optional): e.g., user review history, item categories.
6. Save `edge_index`, node features, and label maps as `.pt` or `.npy`.
   **Deliverables:**

* `edge_index.pt`
* `features.pt` (if applicable)
* ID mapping dictionaries (`user2idx.json`, `item2idx.json`)

### ☐ Task 3: StoreXDataset Class (`graph_dataset.py`)

**What to Do:**
Create a custom dataset class to load data and feed it into PyTorch Geometric.
**Steps:**

1. Subclass `torch.utils.data.Dataset` or use `InMemoryDataset`.
2. Load `edge_index`, features, and any label data.
3. Return `torch_geometric.data.Data` objects.
   **Deliverables:**

* `StoreXDataset` class
* Example notebook or test script showing data loading

### ☐ Task 4: Sampler Class (`sampler.py`)

**What to Do:**
Create logic for generating positive and negative training edges.
**Steps:**

1. Positive sampling: sample user-item pairs that exist in the graph.
2. Negative sampling: generate random user-item pairs that are not connected.
3. Assign labels: 1 for positive, 0 for negative.
4. Return as edge list and label tensors.
   **Deliverables:**

* `Sampler` class
* Saved training edge list: `train_edges.pt`, `train_labels.pt`

### ☐ Task 5: GraphSAGEModel Class (`model.py`)

**What to Do:**
Define the GNN architecture for learning embeddings and predicting interactions.
**Steps:**

1. Define the model using `torch_geometric.nn.SAGEConv` layers.
2. Use optional dropout, batch normalization.
3. Define `forward()` to return node embeddings and/or link prediction logits.
4. Include a `predict(user_id, item_id)` method.
   **Deliverables:**

* `GraphSAGEModel` class
* Unit tests verifying shape and forward pass

### ☐ Task 6: Trainer Class (`trainer.py`)

**What to Do:**
Handle model training and checkpointing.
**Steps:**

1. Load training data from `StoreXDataset` and `Sampler`.
2. Initialize the GNN model, optimizer, and loss function.
3. Train over several epochs with evaluation checkpoints.
4. Save model weights periodically and after best validation score.
   **Deliverables:**

* `Trainer` class
* Trained model file `model_best.pth`
* Training logs (loss, accuracy)

### ☐ Task 7: Evaluator Class (`evaluator.py`)

**What to Do:**
Evaluate top-K recommendation performance.
**Steps:**

1. For each user, rank all candidate items.
2. Use metrics like Precision\@K, Recall\@K, NDCG\@K.
3. Run on held-out test set.
   **Deliverables:**

* Evaluation metrics log file
* `Evaluator` class

### ☐ Task 8: Inference Class (`inference.py`)

**What to Do:**
Generate top-K item recommendations per user.
**Steps:**

1. Load trained model and `StoreXDataset`.
2. Run inference to compute node embeddings.
3. For each user, compute dot product with item embeddings.
4. Rank items and return top-K for each user.
   **Deliverables:**

* User → Top-K item recommendations JSON
* `inference.py` callable script

### ☐ Task 9: Recommender Class (`recommender.py`)

**What to Do:**
High-level orchestration class that binds everything.
**Steps:**

1. Use this class to call GraphBuilder, Trainer, Evaluator, and Inference.
2. Provide two main methods:

   * `run_all()` – runs the pipeline end to end
   * `recommend_top_k(user_id, k)` – returns top K items for API
     **Deliverables:**

* `Recommender` class
* Usable from API or CLI

---

## PHASE 3: Testing & QA

### ☐ Task 10: Unit Testing

**What to Do:**
Write unit tests for robustness and reliability.
**Steps:**

1. Use `pytest` or `unittest`.
2. Write test cases for:

   * Model forward pass
   * Dataset loading
   * Sampler logic (correct number and type of samples)
3. Test edge cases and failure scenarios.
   **Deliverables:**

* `test_model.py`, `test_sampler.py`, `test_dataset.py`
* Test logs

---

## PHASE 4: Django Integration

### ☐ Task 11: Django API Setup (`api/`)

**What to Do:**
Expose the recommender to the frontend via a Django REST API.
**Steps:**

1. Set up Django app `recommender_api`.
2. Create endpoint `/recommend/<user_id>/`.
3. Load saved embeddings and run `Recommender.recommend_top_k()`.
4. Return recommended item IDs or metadata.
   **Deliverables:**

* Django view: `views.py`
* URL route: `urls.py`
* Working JSON API response

### ☐ Task 12: Frontend Integration

**What to Do:**
Render recommended items in the UI.
**Steps:**

1. Modify product listing templates or homepage.
2. Use JavaScript (AJAX) to fetch from API.
3. Display recommended items (e.g., carousel, sidebar).
   **Deliverables:**

* Updated HTML templates
* AJAX call to `/recommend/<user_id>/`
* Rendered recommendation block

---

## PHASE 5: Production & Deployment (Optional)

### ☐ Task 13: Logging and Monitoring

**What to Do:**
Track training and runtime behavior for debugging and optimization.
**Steps:**

1. Log training loss, accuracy, metrics with TensorBoard or Weights & Biases.
2. Log API usage statistics.
   **Deliverables:**

* Training log plots
* API access logs

### ☐ Task 14: Dockerization

**What to Do:**
Package the model and Django API for deployment.
**Steps:**

1. Write `Dockerfile` for model + Django project.
2. Test container locally using `docker-compose`.
   **Deliverables:**

* `Dockerfile`
* Working Docker container image

### ☐ Task 15: Inference Caching

**What to Do:**
Reduce latency by caching recommendation results.
**Steps:**

1. Store top-K recommendations in Redis or memory.
2. Use cache as the first layer in API endpoint.
   **Deliverables:**

* Caching logic
* Benchmark of latency improvement

### ☐ Task 16: CI/CD

**What to Do:**
Automate testing, linting, and deployment.
**Steps:**

1. Add GitHub Actions for running tests on push.
2. Deploy Docker image to server or cloud.
   **Deliverables:**

* `.github/workflows/main.yml`
* Live deployment log

---

## Final Deliverables Checklist

* ☑️ Modular GNN pipeline using OOP
* ☑️ Unit-tested data loader, sampler, trainer, and recommender
* ☑️ RESTful API serving top-K recommendations
* ☑️ Integrated recommendations in Django frontend
* ☑️ Optional: Docker, CI/CD, Caching for production
