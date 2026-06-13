## 🛍️ H&M Fashion Recommendation App

> An interactive Streamlit application for personalized fashion recommendations built on the H&M Fashion Dataset.

---

### 🧠 What's Inside

The platform combines four recommendation approaches:

- **Content-Based** — Sentence-Transformer embeddings for semantic similarity
- **Collaborative Filtering** — ALS model trained on purchase behavior
- **Hybrid** — Combines both signals for better personalization
- **Product Similarity Search** — Find visually and semantically similar items

---

### 📱 Application Modules

**Dashboard**
Project overview, dataset statistics, model architecture summary, and recommendation pipeline explanation.

**Style Discovery**
Cold-start system for new users. Select products you like → system builds a preference embedding → returns semantically similar recommendations.

**Find Similar Styles**
Select any product and discover similar fashion items using cosine similarity on embedding vectors.

**Personalized Picks**
ALS collaborative filtering recommendations based on learned customer interaction patterns and purchase history.

**AI Fashion Advisor**
Hybrid engine combining content and collaborative signals:

```
Final Score = 0.6 × Content Score + 0.4 × ALS Score
```

---

### 📁 Required Model Files

Place the following inside the `models/` directory before running:

```
models/
├── als_model.pkl
├── product_embeddings.pkl
├── user_item_matrix.pkl
├── interaction.pkl
├── article_subset.pkl
├── user2idx.pkl
├── idx2item.pkl
├── item2idx.pkl
├── user_ids.pkl
├── item_ids.pkl
├── article_to_embedding_idx.pkl
└── embedding_idx_to_article.pkl
```

Download instructions → **[download_models.md](https://github.com/Nick-ay/Fashion-Product-Recommendation-Engine/blob/main/models/download_models.md)**

---

### 🚀 Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the app:

```bash
streamlit run app.py
```

Open in browser: `http://localhost:8501`

---

### 🛠️ Tech Stack

`Python` `Streamlit` `Pandas` `NumPy` `Scikit-Learn` `Sentence-Transformers` `Implicit ALS` `SciPy`

---

### 👩‍💻 Author

**Nikita Sheoran**  
Manipal University Jaipur — AI & Recommendation Systems Project
