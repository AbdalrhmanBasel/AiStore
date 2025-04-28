### **🚀 Project Kickoff: GNN Recommendation System (From Scratch)**  
**Project Lead:** [Abdalrhman Morida]  
**Goal:** Build a production-ready GNN recommendation system for Amazon Electronics data.  
**Approach:** We’ll follow the phased plan below, starting with **Phase 1 (Data & Graph Construction)**.  

---

### **📌 Phase 1: Data & Graph Construction**  
**Objective:** Transform raw Amazon data into a graph structure for GNN training.  

#### **Task 1: Create `user2id` and `item2id` Mappings**  
**What to Do:**  
1. Load `reviews_Electronics_5.json` into a DataFrame.  
2. Extract unique user and product IDs.  
3. Map each to contiguous integers (e.g., `"A2EU2U..."` → `0`, `"B07XKG..."` → `1`).  
**Code:**  
```python
user2id = {u: i for i, u in enumerate(reviews_df['reviewerID'].unique())}  
item2id = {p: i for i, p in enumerate(reviews_df['asin'].unique())}  
```  
**Deliverable:** `user2id.json`, `item2id.json`.  

#### **Task 2: Map Datasets to Integer IDs**  
**What to Do:**  
Replace raw IDs in reviews and metadata with mapped integers.  
**Code:**  
```python
reviews_df['user_id'] = reviews_df['reviewerID'].map(user2id)  
meta_df['item_id'] = meta_df['asin'].map(item2id)  
```  
**Deliverable:** `reviews_mapped.parquet`, `meta_mapped.parquet`.  

#### **Task 3: Build Bipartite Graph**  
**What to Do:**  
1. Create edges from user-item interactions.  
2. Save as a sparse COO matrix.  
**Code:**  
```python
edge_index = torch.tensor([reviews_df['user_id'], reviews_df['item_id']], dtype=torch.long)  
torch.save(edge_index, 'data/processed/edge_index.pt')  
```  
**Deliverable:** `edge_index.pt`.  

#### **Task 4: Extract Item Features**  
**Subtasks:**  
1. **One-Hot Encode Categories**:  
   ```python
   from sklearn.preprocessing import OneHotEncoder  
   encoder = OneHotEncoder().fit(meta_df['category'].apply(lambda x: x[0]))  
   ```  
2. **Title Embeddings (BERT)**:  
   ```python
   title_embeddings = model.encode(meta_df['title'].tolist())  
   ```  
3. **Normalize Price**:  
   ```python
   meta_df['price'] = (meta_df['price'] - meta_df['price'].min()) / (meta_df['price'].max() - meta_df['price'].min())  
   ```  
**Deliverable:** `item_features.pt`.  

#### **Task 5: Extract User Features**  
**What to Do:**  
1. **Interaction Count**:  
   ```python
   user_history = reviews_df.groupby('user_id').size().apply(np.log1p)  
   ```  
2. **Average Rating**:  
   ```python
   user_avg_rating = reviews_df.groupby('user_id')['overall'].mean()  
   ```  
**Deliverable:** `user_features.pt`.  

#### **Task 6: Save Processed Data**  
**What to Do:**  
1. Concatenate user/item features:  
   ```python
   features = torch.cat([user_features, item_features], dim=0)  
   ```  
2. Save all components:  
   ```python
   torch.save(features, 'data/processed/features.pt')  
   pickle.dump({'user2id': user2id, 'item2id': item2id}, open('data/processed/mappings.pkl', 'wb'))  
   ```  
**Deliverables:**  
- `features.pt`  
- `labels.pt` (ratings)  
- `mappings.pkl`  

---

### **🔍 Quality Assurance (Before Proceeding to Phase 2)**  
1. **Validate Graph:**  
   ```python
   assert edge_index.max() < len(features), "Edge indices out of bounds!"  
   ```  
2. **Check Features:**  
   ```python
   assert not torch.isnan(features).any(), "NaN values detected!"  
   ```  

---

### **📅 Next Steps**  
1. **Phase 2 (Model Training)**: Implement GraphSAGE in `gnn_model.py`.  
2. **Phase 3 (Inference)**: Build recommendation logic in `recommender.py`.  

**Blockers?** Let’s sync daily at [Time] to troubleshoot.  

--- 

**Let’s build this!** 🚀  
- [Your Name]