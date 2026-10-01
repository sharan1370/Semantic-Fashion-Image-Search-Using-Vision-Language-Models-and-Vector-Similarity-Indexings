import os
import faiss
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import streamlit as st
# Configuration
st.set_page_config(page_title="Fashion Image Search", layout="wide")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_CSV_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
CLIP_INDEX_PATH = os.path.join(DATA_DIR, "clip_index.faiss")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load data & model
@st.cache_resource
def load_all():
    df = pd.read_csv(CLEANED_CSV_PATH)
    index = faiss.read_index(CLIP_INDEX_PATH)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(DEVICE)
    model.eval()
    return df, index, model, processor

df, index, clip_model, clip_processor = load_all()
st.sidebar.success("✅ Model and Index loaded successfully!")
# Helper functions
def search_by_text(query, top_k=20):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_output = clip_model.get_text_features(**inputs)
    text_features = getattr(text_output, "pooler_output", text_output)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_vec = text_features.cpu().numpy().astype("float32")

    distances, indices = index.search(text_vec, top_k)
    return indices[0], distances[0]
# UI Layout

st.title(" Fashion Image Search Engine")
st.markdown("Search fashion products using natural language — powered by **CLIP + FAISS**")

query = st.text_input("Enter your fashion query:", "red floral dress")
top_k = st.slider("Number of results to show", 5, 50, 20)

if st.button("🔍 Search"):
    indices, _ = search_by_text(query, top_k=top_k)
    st.subheader(f"Results for: *{query}*")

    cols = st.columns(5)
    for i, idx in enumerate(indices):
        image_name = str(df.iloc[idx]["image_path"]).replace("\\", "/").rsplit("/", 1)[-1]
        img_path = os.path.join(IMAGE_DIR, image_name)
        if os.path.exists(img_path):
            with cols[i % 5]:
                st.image(Image.open(img_path), caption=f"Rank #{i+1}", use_container_width=True)



