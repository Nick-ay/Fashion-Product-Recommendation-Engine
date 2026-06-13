import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# ----------------------------------------------------
# 1. PAGE SETUP & THEME CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="H&M Fashion Recommender",
    page_icon=":material/styler:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Elegant CSS for a Luxury Fashion Brand Vibe
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FAF9F6 !important;
        color: #1A1A1A !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #222222;
    }
    [data-testid="stSidebar"] * {
        color: #E5DCD3 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    .sidebar-title {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-align: center;
        margin-bottom: 2rem;
        color: #D4AF37 !important;
        border-bottom: 1px solid #333333;
        padding-bottom: 1rem;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 600 !important;
        color: #1A1A1A !important;
        letter-spacing: 1px;
    }
    h1 {
        font-size: 3rem !important;
        border-bottom: 1px solid #E5DCD3;
        padding-bottom: 0.8rem;
        margin-bottom: 1.5rem;
    }
    h2 {
        font-size: 2rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    /* Premium Metric Styling */
    [data-testid="stMetricValue"] {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #8A6D4D !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        color: #666666 !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #EAE5DD;
        border-radius: 8px;
        padding: 15px 20px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.01);
    }

    /* Subtitle Styling */
    .section-subtitle {
        font-size: 1.1rem;
        color: #666666;
        margin-top: -1rem;
        margin-bottom: 2rem;
        font-style: italic;
    }

    /* Divider */
    hr {
        border-color: #E5DCD3 !important;
    }

    /* Target Streamlit's native sidebar collapse/expand icon to fix text-ligature rendering */
    button[aria-label="Collapse sidebar"] [data-testid="stIconMaterial"],
    button[aria-label="Expand sidebar"] [data-testid="stIconMaterial"],
    div[data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"] {
        font-size: 0px !important;
        color: transparent !important;
        position: relative !important;
        width: 24px !important;
        height: 24px !important;
    }

    button[aria-label="Collapse sidebar"] [data-testid="stIconMaterial"]::before,
    div[data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"]::before {
        content: "«" !important;
        font-size: 20px !important;
        color: #A0A0A0 !important;
        font-family: Arial, sans-serif !important;
        position: absolute !important;
        left: 4px !important;
        top: -2px !important;
    }

    button[aria-label="Expand sidebar"] [data-testid="stIconMaterial"]::before {
        content: "»" !important;
        font-size: 20px !important;
        color: #1A1A1A !important;
        font-family: Arial, sans-serif !important;
        position: absolute !important;
        left: 4px !important;
        top: -2px !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. PATH RESOLUTION & ROBUST LOADING
# ----------------------------------------------------
# Resolve paths relative to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

def get_model_path(filename):
    """Robust resolution of pickled model paths relative to 'models/'."""
    # Look in the root models directory first (standard Streamlit execution from root)
    path_root = os.path.join(root_dir, "models", filename)
    if os.path.exists(path_root):
        return path_root
    
    # Fallback: relative directory directly
    path_relative = os.path.join("models", filename)
    if os.path.exists(path_relative):
        return path_relative

    # Fallback: relative to app.py
    path_local = os.path.join(current_dir, "..", "models", filename)
    if os.path.exists(path_local):
        return path_local

    return os.path.join("models", filename)

# ----------------------------------------------------
# 3. CACHED MODEL LOADERS
# ----------------------------------------------------
@st.cache_resource
def load_embeddings():
    path = get_model_path("product_embeddings.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_articles():
    path = get_model_path("article_subset.pkl")
    with open(path, "rb") as f:
        df = pickle.load(f)
    # Basic cleaning and validation
    df["detail_desc"] = df["detail_desc"].fillna("No description available for this item.")
    df["prod_name"] = df["prod_name"].fillna("Unknown Item")
    df["product_type_name"] = df["product_type_name"].fillna("Accessories")
    df["product_group_name"] = df["product_group_name"].fillna("Unknown Group")
    df["colour_group_name"] = df["colour_group_name"].fillna("Multi")
    return df

@st.cache_data
def load_mappings():
    path_to_idx = get_model_path("article_to_embedding_idx.pkl")
    path_from_idx = get_model_path("embedding_idx_to_article.pkl")
    with open(path_to_idx, "rb") as f:
        to_idx = pickle.load(f)
    with open(path_from_idx, "rb") as f:
        from_idx = pickle.load(f)
    return to_idx, from_idx

@st.cache_resource
def load_als_model():
    path = get_model_path("als_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_user_item_matrix():
    path = get_model_path("user_item_matrix.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_als_mappings():
    path_idx2item = get_model_path("idx2item.pkl")
    path_user2idx = get_model_path("user2idx.pkl")
    path_item2idx = get_model_path("item2idx.pkl")
    path_user_ids = get_model_path("user_ids.pkl")
    path_item_ids = get_model_path("item_ids.pkl")
    
    with open(path_idx2item, "rb") as f:
        idx2item = pickle.load(f)
    with open(path_user2idx, "rb") as f:
        user2idx = pickle.load(f)
    with open(path_item2idx, "rb") as f:
        item2idx = pickle.load(f)
    with open(path_user_ids, "rb") as f:
        user_ids = pickle.load(f)
    with open(path_item_ids, "rb") as f:
        item_ids = pickle.load(f)
        
    return idx2item, user2idx, item2idx, user_ids, item_ids

@st.cache_data
def load_interaction_data():
    path = get_model_path("interaction.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

# Load all assets once with spinner feedback
with st.spinner("Loading Fashion intelligence models..."):
    try:
        articles_df = load_articles()
        embeddings = load_embeddings()
        article_to_embedding_idx, embedding_idx_to_article = load_mappings()
        
        # Load collaborative filtering model files
        als_model = load_als_model()
        user_item_matrix = load_user_item_matrix()
        idx2item, user2idx, item2idx, user_ids, item_ids = load_als_mappings()
        interaction_df = load_interaction_data()
    except Exception as e:
        st.error(f"Failed to load dataset files: {e}")
        st.info("Ensure the pickle files are located in the models/ directory relative to the workspace root.")
        st.stop()

# ----------------------------------------------------
# 4. SIDEBAR NAVIGATION & LOGO (SVG Icon Navigation)
# ----------------------------------------------------
# Read page selection from URL query parameters (or default to Dashboard)
query_params = st.query_params
if "page" in query_params:
    selected_page = query_params["page"]
    if selected_page in ["Dashboard", "Style Discovery", "Find Similar Styles", "Personalized Picks", "AI Fashion Advisor"]:
        st.session_state.current_page = selected_page
    else:
        st.session_state.current_page = "Dashboard"
else:
    st.session_state.current_page = "Dashboard"

page = st.session_state.current_page

with st.sidebar:
    # Luxury H&M Custom Logo Monogram (Clean, Gold Accent, Italic Typeface)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.2rem; border-bottom: 1px solid #222; padding-bottom: 1.5rem;">
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 2.3rem; font-weight: 700; color: #D4AF37; letter-spacing: 4px; font-style: italic;">
            H&M
        </div>
        <div style="font-family: 'Outfit', sans-serif; font-size: 0.72rem; letter-spacing: 5px; color: #888888; text-transform: uppercase; margin-top: -6px;">
            STUDIO &bull; LABS
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    
    # Generate active state tags
    active_dash = "active" if page == "Dashboard" else ""
    active_style = "active" if page == "Style Discovery" else ""
    active_find = "active" if page == "Find Similar Styles" else ""
    active_picks = "active" if page == "Personalized Picks" else ""
    active_advisor = "active" if page == "AI Fashion Advisor" else ""
    
    # Premium Vertical HTML/CSS Sidebar using embedded SVG icons (Guarantees zero text rendering & 100% offline-ready)
    st.markdown(f"""
    <style>
        .nav-container {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 1rem;
        }}
        .nav-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            color: #A0A0A0 !important;
            text-decoration: none !important;
            padding: 10px 15px;
            font-family: 'Outfit', sans-serif;
            font-size: 0.95rem;
            font-weight: 400;
            border-radius: 4px;
            border-left: 3px solid transparent;
            transition: all 0.2s ease;
        }}
        .nav-item:hover {{
            color: #E5DCD3 !important;
            background-color: rgba(255, 255, 255, 0.03);
        }}
        .nav-item.active {{
            color: #D4AF37 !important;
            background-color: rgba(212, 175, 55, 0.08);
            border-left: 3px solid #D4AF37;
            font-weight: 600;
        }}
        .nav-icon {{
            flex-shrink: 0;
            stroke: currentColor;
        }}
    </style>
    <div class="nav-container">
        <a href="?page=Dashboard" target="_self" class="nav-item {active_dash}">
            <svg class="nav-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="9"></rect><rect x="14" y="3" width="7" height="5"></rect><rect x="14" y="12" width="7" height="9"></rect><rect x="3" y="16" width="7" height="5"></rect></svg>
            Dashboard
        </a>
        <a href="?page=Style+Discovery" target="_self" class="nav-item {active_style}">
            <svg class="nav-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
            Style Discovery
        </a>
        <a href="?page=Find+Similar+Styles" target="_self" class="nav-item {active_find}">
            <svg class="nav-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            Find Similar Styles
        </a>
        <a href="?page=Personalized+Picks" target="_self" class="nav-item {active_picks}">
            <svg class="nav-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
            Personalized Picks
        </a>
        <a href="?page=AI+Fashion+Advisor" target="_self" class="nav-item {active_advisor}">
            <svg class="nav-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2s-8 6-8 12c0 4.42 3.58 8 8 8s8-3.58 8-8c0-6-8-12-8-12z"></path><path d="M12 10v4"></path></svg>
            AI Fashion Advisor
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="font-family: 'Outfit', sans-serif; font-size: 0.8rem; color: #888888; line-height: 1.5; padding: 0 5px;">
        <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 1.1px; color: #E5DCD3; display: block; margin-bottom: 8px;">Recommendation Engine</span>
        <ul style="list-style-type: none; padding-left: 0; margin-top: 0; margin-bottom: 12px; color: #A0A0A0;">
            <li style="margin-bottom: 4px;">&bull; Semantic Search</li>
            <li style="margin-bottom: 4px;">&bull; Collaborative Filtering (ALS)</li>
            <li style="margin-bottom: 4px;">&bull; Hybrid AI Ranking</li>
        </ul>
        <div style="border-top: 1px solid #222; padding-top: 8px; margin-top: 8px; color: #D4AF37; font-weight: 500; letter-spacing: 0.5px; line-height: 1.6;">
            41K+ Fashion Products<br>
            229K+ Customer Profiles
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# 5. RENDER CHOSEN WORKSPACE
# ----------------------------------------------------

# --- PAGE 1: DASHBOARD ---
if page == "Dashboard":
    st.markdown("<h1>Personalized Fashion Recommender</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">A semantic search and collaborative personalization workbench</div>', unsafe_allow_html=True)
    
    # Display Banner Image if exists
    banner_path = os.path.join(current_dir, "fashion_banner.png")
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)
    
    st.markdown("## Project Overview")
    st.write(
        "Welcome to the H&M Personalized Fashion Recommendation portal. This application leverages "
        "advanced neural representation learning to provide immediate, context-aware styling advice. "
        "By parsing high-dimensional text embeddings generated from H&M's detailed article catalogs, "
        "the recommendation engine captures fine-grained attributes like product names, fabric groups, "
        "garment styles, and color combinations."
    )
    
    st.markdown("---")
    st.markdown("## Dataset & Evaluation Metrics")
    
    # Grid of actual project statistics
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Active Catalog Items", value="41,194")
    col2.metric(label="Registered Users", value="229,536")
    col3.metric(label="Transaction Interactions", value="574,643")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    col4.metric(label="Model Precision @ 10", value="0.0082")
    col5.metric(label="Model Recall @ 10", value="0.0082")
    col6.metric(label="NDCG @ 10 (Model Quality)", value="0.0044")
    
    st.markdown(
        "<div style='font-size: 0.85rem; color: #666666; font-style: italic; margin-top: 15px; text-align: center;'>"
        "Metrics were computed on a highly sparse implicit-feedback fashion dataset containing 229K+ users and 41K+ products."
        "</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown("## Model & Architecture Information")
    st.write(
        "The H&M Personalization engine relies on a multi-stage retrieval architecture designed for "
        "both cold-start scenarios and structured customer journeys:"
    )
    
    # Info tabs
    tab1, tab2 = st.tabs(["Text Representation & Embeddings", "Collaborative Filtering Baseline"])
    with tab1:
        st.markdown("### Sentence-Transformers Embedding Pipeline")
        st.markdown(
            "Every item in the catalog is embedded into a **384-dimensional continuous space** using the "
            "`all-MiniLM-L6-v2` transformer model. The input text represents a descriptive metadata composition:\n"
            "```python\n"
            "text = f\"{prod_name} {product_type_name} {product_group_name} {colour_group_name} {detail_desc}\"\n"
            "```\n"
            "This maps styling semantics, color palettes, and textual patterns directly into vector clusters. "
            "Cosine similarities can then be computed instantenously to return items that align with aesthetic properties."
        )
    with tab2:
        st.markdown("### Alternating Least Squares (ALS) CF Model")
        st.markdown(
            "For active users, an implicit collaborative filtering model (built with `implicit.als.AlternatingLeastSquares`) "
            "is trained on the user-item interaction sparse matrix (229,536 users $\\times$ 41,194 items). The model captures "
            "latent purchase preferences across 64 factors. In production, this collaborative baseline is hybridized "
            "with content embeddings to balance user habit learning with item similarity recommendations."
        )
        
    st.markdown("---")
    st.markdown("# Recommendation Engine")
    
    # 3 Elegant Info Cards
    col_card1, col_card2, col_card3 = st.columns(3)
    
    card1_html = """
    <div style="
        background-color: white; 
        border: 1px solid #EAE5DD; 
        border-radius: 8px; 
        padding: 20px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
        height: 220px;
    ">
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.35rem; font-weight: 700; color: #1A1A1A; margin-bottom: 12px; border-bottom: 1px solid #F3EDE4; padding-bottom: 8px;">
            Semantic Recommendation Engine
        </div>
        <ul style="font-family: 'Outfit', sans-serif; font-size: 0.85rem; color: #555555; line-height: 1.5; padding-left: 15px; margin: 0; list-style-type: disc;">
            <li>Sentence-Transformers (all-MiniLM-L6-v2)</li>
            <li>384-Dimensional Product Embeddings</li>
            <li>Cosine Similarity Retrieval</li>
            <li>Cold-Start Recommendation Support</li>
            <li>Content-Based Ranking</li>
        </ul>
    </div>
    """
    
    card2_html = """
    <div style="
        background-color: white; 
        border: 1px solid #EAE5DD; 
        border-radius: 8px; 
        padding: 20px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
        height: 220px;
    ">
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.35rem; font-weight: 700; color: #1A1A1A; margin-bottom: 12px; border-bottom: 1px solid #F3EDE4; padding-bottom: 8px;">
            Collaborative Recommendation Engine
        </div>
        <ul style="font-family: 'Outfit', sans-serif; font-size: 0.85rem; color: #555555; line-height: 1.5; padding-left: 15px; margin: 0; list-style-type: disc;">
            <li>Alternating Least Squares (ALS)</li>
            <li>Implicit Feedback Learning</li>
            <li>229K+ Customer Profiles</li>
            <li>Behavioral Preference Modeling</li>
            <li>Personalized Recommendation Generation</li>
        </ul>
    </div>
    """
    
    card3_html = """
    <div style="
        background-color: white; 
        border: 1px solid #EAE5DD; 
        border-radius: 8px; 
        padding: 20px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
        height: 220px;
    ">
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.35rem; font-weight: 700; color: #1A1A1A; margin-bottom: 12px; border-bottom: 1px solid #F3EDE4; padding-bottom: 8px;">
            Hybrid Recommendation Engine
        </div>
        <ul style="font-family: 'Outfit', sans-serif; font-size: 0.85rem; color: #555555; line-height: 1.5; padding-left: 15px; margin: 0; list-style-type: disc;">
            <li>Combines Semantic + Collaborative Signals</li>
            <li>60% Content Similarity Weight</li>
            <li>40% ALS Collaborative Weight</li>
            <li>Score Normalization</li>
            <li>Intelligent Recommendation Fusion</li>
        </ul>
    </div>
    """
    
    col_card1.html(card1_html)
    col_card2.html(card2_html)
    col_card3.html(card3_html)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Recommendation Pipeline")
    
    # Visually elegant architecture block flow diagram (100% offline responsive HTML/CSS)
    architecture_html = """
    <div style="
        background-color: white; 
        border: 1px solid #EAE5DD; 
        border-radius: 8px; 
        padding: 25px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
        font-family: 'Outfit', sans-serif;
        margin-top: 5px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: stretch; flex-wrap: wrap; gap: 15px; position: relative;">
            
            <!-- Left Column: Content Pipeline -->
            <div style="flex: 1; min-width: 240px; display: flex; flex-direction: column; align-items: center; justify-content: space-between; gap: 8px; border: 1px solid #FAF9F6; background-color: #FAF9F6; border-radius: 6px; padding: 15px;">
                <div style="font-weight: 700; text-transform: uppercase; font-size: 0.75rem; color: #8A6D4D; letter-spacing: 1.2px; text-align: center; border-bottom: 1px solid #EAE5DD; padding-bottom: 5px; width: 100%; margin-bottom: 8px;">Semantic Pathway (60%)</div>
                
                <div style="background-color: white; border: 1px solid #EAE5DD; border-radius: 4px; padding: 10px; width: 95%; text-align: center; font-size: 0.85rem; font-weight: 500; color: #1A1A1A;">Product Metadata</div>
                <div style="color: #8A6D4D; font-size: 1.1rem; font-weight: 700; line-height: 1;">&darr;</div>
                <div style="background-color: white; border: 1px solid #EAE5DD; border-radius: 4px; padding: 10px; width: 95%; text-align: center; font-size: 0.85rem; font-weight: 500; color: #1A1A1A;">Sentence Transformer</div>
                <div style="color: #8A6D4D; font-size: 1.1rem; font-weight: 700; line-height: 1;">&darr;</div>
                <div style="background-color: white; border: 1px solid #EAE5DD; border-radius: 4px; padding: 10px; width: 95%; text-align: center; font-size: 0.85rem; font-weight: 500; color: #1A1A1A;">Semantic Embeddings</div>
                <div style="color: #8A6D4D; font-size: 1.1rem; font-weight: 700; line-height: 1;">&darr;</div>
                <div style="background-color: white; border: 1px solid #EAE5DD; border-radius: 4px; padding: 10px; width: 95%; text-align: center; font-size: 0.85rem; font-weight: 500; color: #1A1A1A;">Content Recommendations</div>
            </div>
            
            <!-- Center Column Indicator: Fusion Arrow -->
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; width: 40px; color: #8A6D4D; font-size: 1.6rem; font-weight: bold; min-height: 220px;">
                <div>&searr;</div>
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px; color: #888888; font-weight: 600; margin-top: 4px;">60%</div>
            </div>

            <!-- Center Column: Core Hybrid Fusion Node -->
            <div style="flex: 1.1; min-width: 250px; display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #FAF9F6; border: 2px solid #D4AF37; border-radius: 6px; padding: 20px; box-shadow: 0 4px 12px rgba(212,175,55,0.06);">
                <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.6rem; font-weight: 700; color: #1A1A1A; margin-bottom: 6px; text-align: center;">Hybrid Recommendation Engine</div>
                <div style="font-size: 0.7rem; letter-spacing: 2px; color: #8A6D4D; text-transform: uppercase; font-weight: 600; text-align: center; margin-bottom: 12px;">Dynamic Score Fusion</div>
                <div style="font-size: 0.8rem; color: #555555; text-align: center; line-height: 1.45; max-width: 220px;">
                    Normalizes Content similarities and ALS collaborative scoring before executing a weighted linear blending fusion.
                </div>
            </div>

            <!-- Center Column Indicator: Fusion Arrow -->
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; width: 40px; color: #8A6D4D; font-size: 1.6rem; font-weight: bold; min-height: 220px;">
                <div>&nearr;</div>
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px; color: #888888; font-weight: 600; margin-top: 4px;">40%</div>
            </div>

            <!-- Right Column: Collaborative Pipeline -->
            <div style="flex: 1; min-width: 240px; display: flex; flex-direction: column; align-items: center; justify-content: space-between; gap: 8px; border: 1px solid #FAF9F6; background-color: #FAF9F6; border-radius: 6px; padding: 15px;">
                <div style="font-weight: 700; text-transform: uppercase; font-size: 0.75rem; color: #8A6D4D; letter-spacing: 1.2px; text-align: center; border-bottom: 1px solid #EAE5DD; padding-bottom: 5px; width: 100%; margin-bottom: 8px;">Collaborative Pathway (40%)</div>
                
                <div style="background-color: white; border: 1px solid #EAE5DD; border-radius: 4px; padding: 10px; width: 95%; text-align: center; font-size: 0.85rem; font-weight: 500; color: #1A1A1A;">ALS Collaborative Filtering</div>
                <div style="color: #8A6D4D; font-size: 1.1rem; font-weight: 700; line-height: 1;">&uarr;</div>
                <div style="background-color: white; border: 1px solid #EAE5DD; border-radius: 4px; padding: 10px; width: 95%; text-align: center; font-size: 0.85rem; font-weight: 500; color: #1A1A1A;">Customer Interaction History</div>
            </div>
            
        </div>
    </div>
    """
    st.html(architecture_html)


# --- PAGE 2: STYLE DISCOVERY ---
elif page == "Style Discovery":
    st.markdown("<h1>Style Discovery</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Tell us what you like. We will compile a custom styling board for you.</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. Curate Your Inspiration Board")
    st.write("Since you are a new customer, select a few fashion items you find appealing. We will average their visual/semantic representations to tailor custom recommendations.")
    
    # Get unique product names for selection dropdown
    unique_product_names = sorted(articles_df["prod_name"].unique())
    
    selected_names = st.multiselect(
        label="Select products that align with your style preferences:",
        options=unique_product_names,
        default=None,
        placeholder="Type to search e.g. Strap top, Swimwear, Cardigan..."
    )
    
    if not selected_names:
        st.info("Select one or more items above to generate recommendations.")
    else:
        st.markdown("---")
        st.markdown("### 2. Tailored Recommendations for You")
        
        with st.spinner("Analyzing your styling board..."):
            # Get all article IDs matching selected product names
            selected_rows = articles_df[articles_df["prod_name"].isin(selected_names)]
            
            # Map article IDs to embedding indices
            idx_list = []
            for art_id in selected_rows["article_id"]:
                if art_id in article_to_embedding_idx:
                    idx_list.append(article_to_embedding_idx[art_id])
            
            if len(idx_list) == 0:
                st.error("Could not locate embeddings for selected items. Please try choosing other products.")
            else:
                # Calculate the mean embedding of selected items
                selected_vectors = embeddings[idx_list]
                mean_embedding = np.mean(selected_vectors, axis=0).reshape(1, -1)
                
                # Compute cosine similarities between mean embedding and all embeddings
                similarities = cosine_similarity(mean_embedding, embeddings)[0]
                
                # Sort indices in descending order
                sorted_indices = similarities.argsort()[::-1]
                
                # Compile recommendations
                recommendations = []
                seen_names = set(selected_names)  # Exclude user's chosen names
                
                for idx in sorted_indices:
                    item_art_id = embedding_idx_to_article[idx]
                    item_row = articles_df[articles_df["article_id"] == item_art_id]
                    if item_row.empty:
                        continue
                    
                    item_row = item_row.iloc[0]
                    p_name = item_row["prod_name"]
                    
                    # Avoid duplicate product names and selected items
                    if p_name in seen_names:
                        continue
                    
                    seen_names.add(p_name)
                    similarity_percentage = round(float(similarities[idx]) * 100, 1)
                    
                    recommendations.append({
                        "article_id": item_art_id,
                        "prod_name": p_name,
                        "product_type_name": item_row["product_type_name"],
                        "product_group_name": item_row["product_group_name"],
                        "colour_group_name": item_row["colour_group_name"],
                        "detail_desc": item_row["detail_desc"],
                        "score": similarity_percentage
                    })
                    
                    if len(recommendations) >= 10:
                        break
                
                # Display recommendations in a beautiful grid of 2 columns
                cols = st.columns(2)
                for rank, rec in enumerate(recommendations):
                    col = cols[rank % 2]
                    
                    card_html = f"""
                    <div style="
                        background-color: white; 
                        border: 1px solid #EAE5DD; 
                        border-radius: 8px; 
                        padding: 20px; 
                        margin-bottom: 20px; 
                        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
                    ">
                        <div style="text-transform: uppercase; font-size: 10px; font-weight: 600; color: #B8977E; letter-spacing: 1.5px; margin-bottom: 8px;">
                            {rec['product_group_name']} &bull; {rec['product_type_name']}
                        </div>
                        <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.45rem; font-weight: 700; color: #1A1A1A; margin-bottom: 8px; line-height: 1.25;">
                            {rec['prod_name']}
                        </div>
                        <div style="font-size: 12px; color: #666; font-weight: 500; margin-bottom: 8px;">
                            Colorway: <span style="color: #333;">{rec['colour_group_name']}</span>
                        </div>
                        <div style="font-size: 13px; color: #555555; line-height: 1.45; height: 62px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; margin-bottom: 12px;">
                            {rec['detail_desc']}
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #F3EDE4; padding-top: 12px; margin-top: 12px;">
                            <span style="font-size: 11px; color: #999; font-weight: 500;">Ref: #{rec['article_id']}</span>
                            <span style="background-color: #F5EFE6; color: #8A6D4D; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 700;">{rec['score']}% Match</span>
                        </div>
                    </div>
                    """
                    col.html(card_html)


# --- PAGE 3: FIND SIMILAR STYLES ---
elif page == "Find Similar Styles":
    st.markdown("<h1>Find Similar Styles</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Select a product to discover items that share matching aesthetics.</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. Select Query Product")
    unique_product_names = sorted(articles_df["prod_name"].unique())
    
    selected_prod_name = st.selectbox(
        label="Choose a product to find its lookalikes:",
        options=unique_product_names,
        index=0
    )
    
    # Locate article metadata
    matching_articles = articles_df[articles_df["prod_name"] == selected_prod_name]
    
    if matching_articles.empty:
        st.warning("Selected product details could not be found.")
    else:
        st.markdown("---")
        
        # Display details of selected query product in a highlighted callout
        query_item = matching_articles.iloc[0]
        query_art_id = query_item["article_id"]
        
        # Render source card
        st.markdown("### Selected Item Profile")
        st.html(f"""
        <div style="
            background-color: #F5EFE6; 
            border: 1px dashed #B8977E; 
            border-radius: 8px; 
            padding: 20px; 
            margin-bottom: 25px;
        ">
            <div style="text-transform: uppercase; font-size: 10px; font-weight: 600; color: #8A6D4D; letter-spacing: 1.5px; margin-bottom: 6px;">
                QUERY ITEM &bull; {query_item['product_group_name']} &bull; {query_item['product_type_name']}
            </div>
            <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.7rem; font-weight: 700; color: #1A1A1A; margin-bottom: 8px;">
                {query_item['prod_name']}
            </div>
            <div style="font-size: 13px; color: #333; font-weight: 500; margin-bottom: 8px;">
                Primary Colorway: <span>{query_item['colour_group_name']}</span> | Ref: #{query_art_id}
            </div>
            <div style="font-size: 13.5px; color: #555555; line-height: 1.45;">
                {query_item['detail_desc']}
            </div>
        </div>
        """)
        
        st.markdown("### Similar Lookalike Items")
        
        if query_art_id not in article_to_embedding_idx:
            st.error("No embedding index found for the selected query item.")
        else:
            with st.spinner("Finding matching catalog items..."):
                query_idx = article_to_embedding_idx[query_art_id]
                query_vector = embeddings[query_idx].reshape(1, -1)
                
                # Calculate cosine similarities
                similarities = cosine_similarity(query_vector, embeddings)[0]
                
                # Sort indices descending
                sorted_indices = similarities.argsort()[::-1]
                
                # Fetch recommendations
                similar_items = []
                seen_names = set([selected_prod_name])
                
                for idx in sorted_indices:
                    item_art_id = embedding_idx_to_article[idx]
                    item_row = articles_df[articles_df["article_id"] == item_art_id]
                    if item_row.empty:
                        continue
                    
                    item_row = item_row.iloc[0]
                    p_name = item_row["prod_name"]
                    
                    # Skip duplicate product names and query product
                    if p_name in seen_names:
                        continue
                    
                    seen_names.add(p_name)
                    similarity_percentage = round(float(similarities[idx]) * 100, 1)
                    
                    similar_items.append({
                        "article_id": item_art_id,
                        "prod_name": p_name,
                        "product_type_name": item_row["product_type_name"],
                        "product_group_name": item_row["product_group_name"],
                        "colour_group_name": item_row["colour_group_name"],
                        "detail_desc": item_row["detail_desc"],
                        "score": similarity_percentage
                    })
                    
                    if len(similar_items) >= 10:
                        break
                
                # Display recommendations in a grid of 2 columns
                cols = st.columns(2)
                for rank, rec in enumerate(similar_items):
                    col = cols[rank % 2]
                    
                    card_html = f"""
                    <div style="
                        background-color: white; 
                        border: 1px solid #EAE5DD; 
                        border-radius: 8px; 
                        padding: 20px; 
                        margin-bottom: 20px; 
                        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
                    ">
                        <div style="text-transform: uppercase; font-size: 10px; font-weight: 600; color: #B8977E; letter-spacing: 1.5px; margin-bottom: 8px;">
                            {rec['product_group_name']} &bull; {rec['product_type_name']}
                        </div>
                        <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.45rem; font-weight: 700; color: #1A1A1A; margin-bottom: 8px; line-height: 1.25;">
                            {rec['prod_name']}
                        </div>
                        <div style="font-size: 12px; color: #666; font-weight: 500; margin-bottom: 8px;">
                            Colorway: <span style="color: #333;">{rec['colour_group_name']}</span>
                        </div>
                        <div style="font-size: 13px; color: #555555; line-height: 1.45; height: 62px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; margin-bottom: 12px;">
                            {rec['detail_desc']}
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #F3EDE4; padding-top: 12px; margin-top: 12px;">
                            <span style="font-size: 11px; color: #999; font-weight: 500;">Ref: #{rec['article_id']}</span>
                            <span style="background-color: #F5EFE6; color: #8A6D4D; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 700;">{rec['score']}% Match</span>
                        </div>
                    </div>
                    """
                    col.html(card_html)


# --- PAGE 4: PERSONALIZED PICKS ---
elif page == "Personalized Picks":
    st.markdown("<h1>Personalized Picks</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Collaborative Filtering recommendation based on user transaction profiles.</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. Identify Customer Profile")
    user_idx = st.number_input(
        label="Select Customer Index (0 to 229,535):",
        min_value=0,
        max_value=229535,
        value=0,
        step=1
    )
    
    # Retrieve customer original ID and purchases
    original_user_id = user_ids[user_idx]
    user_interactions = interaction_df[interaction_df["user_idx"] == user_idx]
    total_interactions = int(user_interactions["count"].sum())
    
    # Display statistics for context
    col1, col2, col3 = st.columns(3)
    col1.metric(label="User Index Selection", value=str(user_idx))
    col2.metric(label="Total Interaction Units", value=str(total_interactions))
    col3.metric(label="Unique Articles Purchased", value=str(len(user_interactions)))
    
    st.markdown(f"**Original H&M Customer ID Hash:** `{original_user_id}`")
    st.markdown("---")
    
    # Purchase History Log section
    st.markdown("### Purchase History Log")
    if user_interactions.empty:
        st.info("No transaction interactions recorded for this customer index in the sample subset.")
    else:
        history_items = []
        for _, row in user_interactions.iterrows():
            item_idx = int(row["item_idx"])
            count = int(row["count"])
            
            # Map index to article_id gracefully
            if item_idx in idx2item:
                art_id = idx2item[item_idx]
                matching_rows = articles_df[articles_df["article_id"] == art_id]
                if not matching_rows.empty:
                    art_row = matching_rows.iloc[0]
                    history_items.append({
                        "prod_name": art_row["prod_name"],
                        "product_group_name": art_row["product_group_name"],
                        "product_type_name": art_row["product_type_name"],
                        "colour_group_name": art_row["colour_group_name"],
                        "count": count,
                        "article_id": art_id
                    })
        
        if history_items:
            history_table = pd.DataFrame(history_items)
            st.dataframe(
                history_table[["prod_name", "product_group_name", "product_type_name", "colour_group_name", "count", "article_id"]],
                use_container_width=True,
                column_config={
                    "prod_name": "Product Name",
                    "product_group_name": "Product Category Group",
                    "product_type_name": "Garment Type",
                    "colour_group_name": "Color Scheme",
                    "count": "Purchase Count",
                    "article_id": "Article Reference ID"
                }
            )
        else:
            st.info("Interaction indexes are recorded, but corresponding product meta lies outside the catalog subset.")

    st.markdown("---")
    st.markdown("### Collaborative Recommendations")
    
    # ALS Recommendation calculations
    with st.spinner("Analyzing customer purchase vectors and retrieving ALS items..."):
        try:
            # Generate top 10 collaborative recommendations
            rec_ids, rec_scores = als_model.recommend(
                userid=user_idx,
                user_items=user_item_matrix[user_idx],
                N=10,
                filter_already_liked_items=True
            )
            
            # Render recommendations in 2-column cards grid
            cols = st.columns(2)
            displayed_recs = 0
            
            for rank, (item_idx, score) in enumerate(zip(rec_ids, rec_scores)):
                # Convert ALS item index back to article ID string
                if item_idx not in idx2item:
                    continue
                
                art_id = idx2item[item_idx]
                
                # Fetch metadata
                matching_rows = articles_df[articles_df["article_id"] == art_id]
                if matching_rows.empty:
                    continue
                
                art_row = matching_rows.iloc[0]
                col = cols[displayed_recs % 2]
                displayed_recs += 1
                
                card_html = f"""
                <div style="
                    background-color: white; 
                    border: 1px solid #EAE5DD; 
                    border-radius: 8px; 
                    padding: 20px; 
                    margin-bottom: 20px; 
                    box-shadow: 0 4px 10px rgba(0,0,0,0.02);
                ">
                    <div style="text-transform: uppercase; font-size: 10px; font-weight: 600; color: #B8977E; letter-spacing: 1.5px; margin-bottom: 8px;">
                        {art_row['product_group_name']} &bull; {art_row['product_type_name']}
                    </div>
                    <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.45rem; font-weight: 700; color: #1A1A1A; margin-bottom: 8px; line-height: 1.25;">
                        {art_row['prod_name']}
                    </div>
                    <div style="font-size: 12px; color: #666; font-weight: 500; margin-bottom: 8px;">
                        Colorway: <span style="color: #333;">{art_row['colour_group_name']}</span>
                    </div>
                    <div style="font-size: 13px; color: #555555; line-height: 1.45; height: 62px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; margin-bottom: 12px;">
                        {art_row['detail_desc']}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #F3EDE4; padding-top: 12px; margin-top: 12px;">
                        <span style="font-size: 11px; color: #999; font-weight: 500;">Ref: #{art_id}</span>
                        <span style="background-color: #F5EFE6; color: #8A6D4D; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 700;">ALS Score: {score:.5f}</span>
                    </div>
                </div>
                """
                col.html(card_html)
                
            if displayed_recs == 0:
                st.warning("Collaborative recommendations successfully resolved from user history, but the products lie outside the catalog subset.")
                
        except Exception as e:
            st.error(f"Error computing collaborative filtering: {e}")


# --- PAGE 5: AI FASHION ADVISOR ---
elif page == "AI Fashion Advisor":
    st.markdown("<h1>AI Fashion Advisor</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Fused scoring engine blending semantic content similarity (60%) and collaborative ALS latent factors (40%).</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. Choose Product Inspirations")
    unique_product_names = sorted(articles_df["prod_name"].unique())
    
    selected_names = st.multiselect(
        label="Select products to define your target style:",
        options=unique_product_names,
        default=None,
        placeholder="Type to search e.g. Strap top, Swimwear, Cardigan..."
    )
    
    if not selected_names:
        st.info("Select one or more items above to generate hybrid recommendations.")
    else:
        st.markdown("---")
        st.markdown("### 2. Hybrid Recommendation Board")
        
        with st.spinner("Processing style representations and blending model signals..."):
            # Fetch matching article IDs
            selected_rows = articles_df[articles_df["prod_name"].isin(selected_names)]
            selected_art_ids = selected_rows["article_id"].tolist()
            
            # --- CONTENT MODEL (Text Embeddings) ---
            # Retrieve embedding indices for selections
            emb_indices = [
                article_to_embedding_idx[art_id] 
                for art_id in selected_art_ids 
                if art_id in article_to_embedding_idx
            ]
            
            if len(emb_indices) == 0:
                st.error("No embedding coordinates found for selected items.")
            else:
                # Compute averaged user profile embedding
                selected_vectors = embeddings[emb_indices]
                mean_embedding = np.mean(selected_vectors, axis=0).reshape(1, -1)
                
                # Compute raw cosine similarities across all embedding rows
                content_score_raw = cosine_similarity(mean_embedding, embeddings)[0]
                
                # Apply Min-Max Normalization to Content Scores (Scale to [0, 1])
                min_c = content_score_raw.min()
                max_c = content_score_raw.max()
                normalized_content = (content_score_raw - min_c) / (max_c - min_c + 1e-9)
                
                # --- COLLABORATIVE MODEL (ALS Item Latent Factors) ---
                # Retrieve item factors for selections
                als_indices = [
                    item2idx[art_id] 
                    for art_id in selected_art_ids 
                    if art_id in item2idx and item2idx[art_id] < 41194
                ]
                
                if len(als_indices) == 0:
                    # Fallback mapping: Set ALS scores to zero if none of selected items have ALS latent vectors
                    als_score_raw = np.zeros_like(content_score_raw)
                    normalized_als = np.zeros_like(content_score_raw)
                else:
                    # Retrieve and average latent factors from als_model.item_factors
                    selected_factors = als_model.item_factors[als_indices]
                    mean_factor = np.mean(selected_factors, axis=0).reshape(1, -1)
                    
                    # Compute raw cosine similarities across all ALS item factor coordinates
                    als_score_raw = cosine_similarity(mean_factor, als_model.item_factors)[0]
                    
                    # Apply Min-Max Normalization to ALS Scores (Scale to [0, 1])
                    min_a = als_score_raw.min()
                    max_a = als_score_raw.max()
                    normalized_als = (als_score_raw - min_a) / (max_a - min_a + 1e-9)
                
                # --- SCORE FUSION AND META-MAPPING ---
                hybrid_items = []
                
                for idx, row in articles_df.iterrows():
                    art_id = row["article_id"]
                    
                    # Rule 4: Check if embedding index exists. If not, Content Score = 0.
                    emb_idx = article_to_embedding_idx.get(art_id)
                    c_score = float(normalized_content[emb_idx]) if emb_idx is not None else 0.0
                    
                    # Rule 3: Check if ALS index mapping exists. If not, ALS Score = 0.
                    als_idx = item2idx.get(art_id)
                    a_score = float(normalized_als[als_idx]) if (als_idx is not None and als_idx < 41194) else 0.0
                    
                    # Fusion calculation: final_score = 0.6 * normalized_content + 0.4 * normalized_als
                    final_score = 0.6 * c_score + 0.4 * a_score
                    
                    hybrid_items.append({
                        "article_id": art_id,
                        "prod_name": row["prod_name"],
                        "product_type_name": row["product_type_name"],
                        "product_group_name": row["product_group_name"],
                        "colour_group_name": row["colour_group_name"],
                        "detail_desc": row["detail_desc"],
                        "final_score": final_score,
                        "content_score": c_score,
                        "als_score": a_score
                    })
                
                # Convert list to DataFrame to filter and sort
                hybrid_df = pd.DataFrame(hybrid_items)
                
                # Rule 1: Exclude product names selected by the user
                hybrid_df = hybrid_df[~hybrid_df["prod_name"].isin(selected_names)]
                
                # Sort by blended final score descending
                hybrid_df = hybrid_df.sort_values(by="final_score", ascending=False)
                
                # Rule 2: Remove duplicate product names, keeping highest score
                hybrid_df = hybrid_df.drop_duplicates(subset=["prod_name"], keep="first")
                
                # Rule 5: Retrieve top 10 items
                top_10_recs = hybrid_df.head(10)
                
                # Display recommendations in a beautiful grid of 2 columns
                cols = st.columns(2)
                for rank, (_, rec) in enumerate(top_10_recs.iterrows()):
                    col = cols[rank % 2]
                    
                    # Format scores as percentages
                    final_pct = round(rec["final_score"] * 100, 1)
                    content_pct = round(rec["content_score"] * 100, 1)
                    als_pct = round(rec["als_score"] * 100, 1)
                    
                    card_html = f"""
                    <div style="
                        background-color: white; 
                        border: 1px solid #EAE5DD; 
                        border-radius: 8px; 
                        padding: 20px; 
                        margin-bottom: 20px; 
                        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
                    ">
                        <div style="text-transform: uppercase; font-size: 10px; font-weight: 600; color: #B8977E; letter-spacing: 1.5px; margin-bottom: 8px;">
                            {rec['product_group_name']} &bull; {rec['product_type_name']}
                        </div>
                        <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.45rem; font-weight: 700; color: #1A1A1A; margin-bottom: 8px; line-height: 1.25;">
                            {rec['prod_name']}
                        </div>
                        <div style="font-size: 12px; color: #666; font-weight: 500; margin-bottom: 8px;">
                            Colorway: <span style="color: #333;">{rec['colour_group_name']}</span> | Ref: #{rec['article_id']}
                        </div>
                        <div style="font-size: 13px; color: #555555; line-height: 1.45; height: 62px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; margin-bottom: 12px;">
                            {rec['detail_desc']}
                        </div>
                        
                        <div style="background-color: #FAF9F6; border-radius: 6px; padding: 12px; border: 1px solid #F3EDE4; margin-top: 12px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <span style="font-size: 12.5px; font-weight: 700; color: #1A1A1A;">Final Score</span>
                                <span style="font-size: 12.5px; font-weight: 700; color: #8A6D4D;">{final_pct}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: #666666; margin-bottom: 2px;">
                                <span>Content Score (60%)</span>
                                <span>{content_pct}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: #666666;">
                                <span>ALS Score (40%)</span>
                                <span>{als_pct}%</span>
                            </div>
                        </div>
                    </div>
                    """
                    col.html(card_html)
