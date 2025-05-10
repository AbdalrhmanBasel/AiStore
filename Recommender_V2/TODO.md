TODO:

## Step #1: Data Cleaning
- [x] Encode/parse the `parent_asin` in `meta_df` & `review_df` into normal IDs start from zero under name `product_id`.
- [x] Encode/parse the `user_id` in the `reviews_df` into normal IDs start from zero.
- [x] Save newly parsed `meta_df` and `reviews_df` datasets.

## Step #2: Graph Building
- [x] Load the parsed `meta_df` and `reviews_df` datasets.
- [x] Build a bipartite graph that includes **product_numerical_features** and **weighted edges** from `reviews_df['ratings'].
- [x] Using `RandomLinkSplit`, split the links from the graph for performing Link Prediction training
- [x] Save the bipartite `full_graph.pt`, `train_graph.pt`, `val_graph.pt`, and `test_graph.pt` in the directory `artifacts/data`.

## Step #3: Training Model
- [x] Load  `full_graph.pt`, `train_graph.pt`, `val_graph.pt`, and `test_graph.pt` from  directory `artifacts/data`.
- [x] Build `HeteroGraphSAGE` model to accept `user`, `rates`, `item` nodes and edges.
- [x] Build `LinkPredictor` model to predict links.
- [x] Build `training` function with hyperparemeter tuning and schaduler.
- [x] Build `evaluate` function to evaluate model performance.
- [x] Train `HeteroGraphSAGE` model and `LinkPredictor` model.
- [x] Save `HeteroGraphSAGE` model and `LinkPredictor` model to `artifacts/models` directory.

## Step #4: Recommending Product IDs.
- [x] Load `full_graph.pt`, `train_graph.pt`, `val_graph.pt`, and `test_graph.pt` from  directory `artifacts/data`.
- [x] Load `HeteroGraphSAGE` model and `LinkPredictor` model to `artifacts/models` from the directory.
- [x] Build `top_k_recommender` function to recommend top-k products based on `user_id` input.
- [x] Build `similar_items_recommender`.function to recommend similar products based on `product_ids`.

## Step #5: Setting Up PostgreSQL Database.
- [] Integrate database PostgreSQL into Django web service.
- [ ] Design Django Database schema that includes `users`, `products` models.
- [ ] Implement `Users` model to include user attributes such as `user_id`, `first_name`, `last_name`, `age`, `gender`, `address`.
- [ ] Implement `Reviews` model to include: `user_id`, `product_id`, `rating`, `timestamp`, `year`, `month`, `hour`, and `minute` attributes.
- [ ] Implement `Products` model to include:`product_id`, `main_category`, `title`, `averate_rating`, `description`, `price`, `store`, `brand`, `color`, `date_first_available`, `primary_category`, amd `rating_bin`.
- [ ] Implement `Categories` model to include: `category_id`, `parent_category`, `title`, and`description`.
- [ ] Implement `orders` model to include: `order_id`, `user_id`, `product_id`
- [ ] Upload `reviews_df` and `meta_df` to PostgreSQL database.

## Step #6: Integrating Recommendation System Into Django AiStore Web Service.
- [ ] Build Recommender Django application.
- [ ] Write `graph_loader` function to load graph.
- [ ] Write `HeteroGraphSAGE` model and `LinkPredictor` model to load models.
- [ ] Write `recommend_top_k` function in `views.py` to send product api data based on the given list of `product_id`.

## Step #7: Develop Ecommerce Website Frontned
- [ ] Develop Navbar frontend.
- [ ] Develop Footer frontend.
- [ ] Develop HomePage frontend & list random products.
- [ ] Develop checkout frontend page.
- [ ] Develop cart frontend page.
