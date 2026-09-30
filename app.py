import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="TF-IDF Explorer", layout="wide")
st.title("Data centers in Argentina — TF-IDF explorer")

SEG_FILE = "tfidf_by_segment.csv"
TERMS_FILE = "tf_terms.csv"


@st.cache_data
def load_data():
    seg = pd.read_csv(SEG_FILE) if os.path.exists(SEG_FILE) else pd.DataFrame()
    if not seg.empty:
        seg = seg.rename(columns={"source_type": "segment", "mean_tfidf": "tfidf_score"})
        seg["term"] = seg["term"].astype(str).str.replace("_", " ")
    terms = pd.read_csv(TERMS_FILE) if os.path.exists(TERMS_FILE) else pd.DataFrame()
    if not terms.empty:
        terms["term"] = terms["term"].astype(str).str.replace("_", " ")
    return seg, terms


segment_df, terms_df = load_data()

if segment_df.empty:
    st.error(f"`{SEG_FILE}` not found next to app.py. Files here: {sorted(os.listdir('.'))}")
    st.stop()

required = {"segment", "term", "tfidf_score"}
if not required.issubset(segment_df.columns):
    st.error(f"`{SEG_FILE}` has columns {list(segment_df.columns)}; expected {sorted(required)}")
    st.stop()

# ---------------- sidebar ----------------
st.sidebar.header("Controls")
top_n = st.sidebar.slider("Top N terms", 5, 50, 15)
segments = sorted(segment_df["segment"].astype(str).unique())
selected = st.sidebar.selectbox("Segment", segments)

# ---------------- 1. bar chart ----------------
st.header("1. Most characteristic terms per segment")
sub = (
    segment_df[segment_df["segment"].astype(str) == selected]
    .sort_values("tfidf_score", ascending=False)
    .head(top_n)
)
n_docs = int(sub["n_docs"].iloc[0]) if "n_docs" in sub.columns and len(sub) else None
title = selected + (f"  (n = {n_docs} tweets)" if n_docs else "")

fig_bar = px.bar(
    sub,
    x="tfidf_score",
    y="term",
    orientation="h",
    color="tfidf_score",
    color_continuous_scale="Blues",
    labels={"tfidf_score": "Mean TF-IDF weight (0–1)", "term": ""},
    title=title,
)
fig_bar.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=550,
    coloraxis_showscale=False,
)
st.plotly_chart(fig_bar, width="stretch")

# ---------------- 2. heatmap ----------------
st.header("2. Term × segment heatmap")
pivot = segment_df.pivot_table(index="term", columns="segment", values="tfidf_score", fill_value=0)
top_terms = segment_df.groupby("term")["tfidf_score"].sum().nlargest(top_n).index
fig_heat = px.imshow(
    pivot.loc[pivot.index.isin(top_terms)],
    labels=dict(x="Segment", y="Term", color="Mean TF-IDF"),
    color_continuous_scale="Viridis",
    aspect="auto",
    title=f"Top {top_n} terms across segments",
)
fig_heat.update_layout(height=max(400, 22 * top_n))
st.plotly_chart(fig_heat, width="stretch")
st.caption("Only terms in a segment's top-20 are stored, so most cells are 0 by construction.")

# ---------------- 3. lookup ----------------
st.header("3. Term frequency lookup")
query = st.text_input("Filter terms", "")
if terms_df.empty:
    st.info(f"`{TERMS_FILE}` not found — lookup disabled.")
else:
    view = (
        terms_df[terms_df["term"].str.contains(query, case=False, na=False)]
        if query
        else terms_df.head(top_n)
    )
    st.dataframe(view, width="stretch", hide_index=True)
