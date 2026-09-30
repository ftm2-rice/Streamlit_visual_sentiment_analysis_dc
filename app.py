import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="TF-IDF Interactive Explorer", layout="wide")
st.title("📊 Interactive TF-IDF & Text Segment Explorer")

SEG_FILE = "tfidf_by_segment.csv"     # <- change if your files are named *_2.csv
TERMS_FILE = "tf_terms.csv"

@st.cache_data
def load_data():
    seg = pd.read_csv(SEG_FILE) if os.path.exists(SEG_FILE) else pd.DataFrame()
    if not seg.empty:
        seg = seg.rename(columns={"source_type": "segment", "mean_tfidf": "tfidf_score"})
        seg["term"] = seg["term"].str.replace("_", " ")
    terms = pd.read_csv(TERMS_FILE) if os.path.exists(TERMS_FILE) else pd.DataFrame()
    if not terms.empty:
        terms["term"] = terms["term"].str.replace("_", " ")
    return seg, terms

segment_df, terms_df = load_data()

if segment_df.empty:
    st.error(f"`{SEG_FILE}` not found next to app.py. Files in folder: {os.listdir('.')}")
    st.stop()

st.sidebar.header("Global Controls")
top_n = st.sidebar.slider("Top N terms", 5, 50, 15)
segments = sorted(segment_df["segment"].unique())
selected = st.sidebar.selectbox("Segment", segments)

# 1. Bar chart
st.header("1. Most characteristic terms per segment")
sub = (segment_df[segment_df["segment"] == selected]
       .sort_values("tfidf_score", ascending=False).head(top_n))
n = int(sub["n_docs"].iloc[0]) if "n_docs" in sub else None
fig_bar
