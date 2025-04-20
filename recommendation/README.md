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
- [ ] Create `user2id` and `item2id` mappings for users and items
- [ ] Map all datasets to integer IDs (users, items, etc.)
- [ ] Build bipartite graph (`user ↔ item`) based on user-item interactions
- [ ] Extract and preprocess item features:
  - [ ] One-hot encode `category`
  - [ ] Embed `title` using TF-IDF or BERT
  - [ ] Normalize `price` and encode `brand`
- [ ] Extract user features:
  - [ ] Calculate user history size (number of interactions)
  - [ ] Calculate average rating for each user
- [ ] Save processed data:
  - [ ] `edge_index.pt`: Graph structure (user-item edges)
  - [ ] `features.pt`: Node features (item/user features)
  - [ ] `labels.pt`: Target labels (interactions, ratings)
  - [ ] `mappings.pkl`: Mappings for user and item IDs

## 🧠 PHASE 2: Model Training (GraphSAGE)

- [ ] Implement GraphSAGE model in `gnn_model.py`
- [ ] Use `NeighborLoader` from PyTorch Geometric for mini-batch training
- [ ] Define scoring head (dot product or MLP)
- [ ] Define loss function (binary cross-entropy with negative sampling)
- [ ] Add evaluation metrics:
  - [ ] Precision@K
  - [ ] Recall@K
  - [ ] NDCG@K (Normalized Discounted Cumulative Gain)
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


## 🔧 Folder Structure

The folder structure is organized for **modularity**, **scalability**, and **maintainability**. Each folder and file is dedicated to specific functionalities within the recommendation system development pipeline. Below is a detailed overview of the folder structure:

```bash
recommendation/
│
├── config                           # Configuration files for model and training settings
│   ├── base_config.yaml             # Base configuration settings (common across all models)
│   ├── model_config.yaml            # Model-specific configuration (layers, dimensions)
│   └── train_config.yaml            # Training settings (learning rate, batch size, epochs)
│
├── data                             # Data-related files (raw, processed, interim data)
│   ├── interim                      # Files for interim/preprocessed data
│   ├── processed                    # Processed data used for model training and evaluation
│   └── raw                          # Raw data files (Amazon dataset)
│
│
├── experiments                      # Experiment files for debugging, baseline comparisons, and studies
│   ├── ablation_study.ipynb         # Jupyter notebook for model ablation studies
│   ├── baseline_train.py            # Script to train baseline models
│   └── debug_graph.ipynb            # Jupyter notebook for debugging graph construction
│
├── pyproject.toml                   # Python project configuration file (dependencies, etc.)
├── README.md                        # Project overview and instructions
├── requirements.txt                 # Project dependencies (Python libraries)
│
├── scripts                          # Standalone scripts for various tasks
│   ├── preprocess_data.py           # Script to preprocess raw data for training
│   ├── train_model.py               # Script to train the recommendation model
│   └── visualize_embeddings.py      # Script for visualizing node embeddings
│
├── setup.py                         # Setup file for project installation
│
├── src                           # Main project code for the recommendation engine
│   ├── data                         # Data processing, graph construction, and dataset
│   │   ├── feature_engineering.py   # Feature engineering for item and user data
│   │   ├── graph_constructor.py     # Logic for constructing the recommendation graph
│   │   ├── graph_dataset.py         # Dataset class for handling graph data
│   │   └── preprocessing.py         # Preprocessing logic (text cleaning, feature extraction)
│   ├── models                       # Model architecture files
│   │   ├── encoders.py              # Neural network layers and encoder logic
│   │   ├── graphsage.py             # Implementation of the GraphSAGE model
│   │   ├── __init__.py              # Model initialization script
│   │   └── utils.py                 # Utility functions for model operations
│   ├── training                     # Training-related scripts
│   │   ├── callbacks.py             # Callbacks for training (early stopping)
│   │   ├── eval.py                  # Evaluation functions for model performance
│   │   ├── __init__.py              # Training package initialization
│   │   ├── loss.py                  # Custom loss functions (binary cross-entropy)
│   │   └── train.py                 # Main training loop
│   ├── utils                        # Utility functions for logging, configuration, etc.
│   │   ├── config_loader.py         # Configuration file loader
│   │   ├── logger.py                # Logger for debugging and tracking
│   │   ├── metrics.py               # Evaluation metrics for performance (Precision, Recall)
│   │   └── seed.py                  # Seed initialization for reproducibility
│   └── visualization                # Visualization tools for graph embeddings and model performance
│       ├── graph_plotter.py         # Graph plotting utility
│       └── tsne_embeddings.py       # t-SNE visualization of node embeddings
│
└── tests                            # Unit and integration tests
    ├── test_graph_dataset.py        # Unit tests for graph dataset
    ├── test_model.py                # Unit tests for model code
    ├── test_preprocessing.py        # Unit tests for data preprocessing
    └── test_train_loop.py           # Unit tests for training process

```
---

## **Installation**

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/storex-gnn-rec-sys.git
    cd storex-gnn-rec-sys
    ```

2. **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Set up the environment (Optional):**
    - Create your own `.env` file if necessary for API keys, model parameters, etc.

---

## **Data Preprocessing**

To begin processing the raw data and constructing the graph, run the following script:

```bash
python scripts/preprocess_data.py
```

This will:
- Download the raw Amazon Electronics dataset.
- Clean the data and generate the user-item mappings and feature matrices.
- Save the processed files in the `data/processed/` directory.


## **Training the Model**

To train the GraphSAGE model, run:

```bash
python scripts/train_model.py
```

This script will:
- Load the preprocessed data.
- Train the model using GraphSAGE and save the model weights.
- Output embeddings for users and items, which will be used for generating recommendations.



## **Visualizing Embeddings**

To visualize the learned embeddings, use the following script:

```bash
python scripts/visualize_embeddings.py
```

This will:
- Use t-SNE or UMAP to reduce the dimensionality of the embeddings and visualize them.



## **Running the Django App**

To integrate the recommendation system into a Django application:

1. **Start the Django server:**
    ```bash
    python manage.py runserver
    ```

2. **Use the API endpoint:**
    - GET `/recommend/<user_id>/`: Returns personalized recommendations for a given user.


## **Testing**

To run unit tests:

```bash
pytest tests/
```

This will test all the core modules, including preprocessing, model training, and the data pipeline.


## **Evaluation**

After training, evaluate the model using the `eval.py` script. This script calculates key metrics like Precision@K, Recall@K, and NDCG@K.

## **Contributions**

Feel free to open issues and pull requests. We welcome contributions to improve the performance, modularity, or scalability of the recommendation system.

## **License**

Distributed under the MIT License. See `LICENSE` for more information.

## **Acknowledgments**

- Amazon for the dataset.
- PyTorch for building GNN models.
- Various libraries used in this project, including `networkx`, `torch`, and `scikit-learn`.



## Author
### Abdalrhman Morida
#### AI Engineer and CEO of Ai Solutions LLC

Abdalrhman is an AI engineer with a strong passion for building scalable AI solutions and revolutionizing AI education. As the CEO of Ai Solutions LLC, he focuses on bringing innovative AI-driven products and services to businesses. With a deep commitment to both technical development and leadership, Abdalrhman aims to transform industries through advanced technologies like Graph Neural Networks (GNNs). He is currently working on various AI projects, including the development of recommendation systems and solutions for the eCommerce sector.

## Contact Author
- **Email:** Abdalrhman30x@gmail.com
- **Telegram:** Abdalrhman30x

