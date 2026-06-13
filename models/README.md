## 📦 Model Files

> Trained model artifacts are stored externally due to GitHub file size limits.  
> Download instructions → **[download_models.md](https://github.com/Nick-ay/Fashion-Product-Recommendation-Engine/blob/main/models/download_models.md)**

After downloading, place all files inside the `models/` directory.

---

### 📁 Directory Structure

```
models/
├── als_model.pkl                  # Trained ALS collaborative filtering model
├── product_embeddings.pkl         # Sentence-Transformer embeddings for all products
├── user_item_matrix.pkl           # Sparse user-item interaction matrix
├── interaction.pkl                # Processed user-product engagement data
├── article_subset.pkl             # H&M product catalog with metadata
├── user2idx.pkl                   # User ID → ALS index
├── idx2item.pkl                   # ALS index → article ID
├── item2idx.pkl                   # Article ID → ALS index
├── user_ids.pkl                   # Unique user identifiers from training
├── item_ids.pkl                   # Unique product identifiers from training
├── article_to_embedding_idx.pkl   # Article ID → embedding index
└── embedding_idx_to_article.pkl   # Embedding index → article ID
```

---

### ✅ Verify Setup

Once all files are in place, run:

```bash
streamlit run app/app.py
```

If the app launches without errors — you're all set.
