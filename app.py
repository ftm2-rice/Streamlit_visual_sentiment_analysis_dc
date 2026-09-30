import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Twitter Data Analysis", layout="wide")
st.title("Part 6: Visualizations Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("outputs/processed_data.csv")
    tf_terms = pd.read_csv("outputs/tf_terms.csv")
    tfidf_seg = pd.read_csv("outputs/tfidf_by_segment.csv")
    return df, tf_terms, tfidf_seg

df, tf_terms, tfidf_seg = load_data()

# Chart 1: Top 25 terms by document frequency
st.header("1. Document Frequency")
top25 = tf_terms.head(25).iloc[::-1]
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.barh(top25["term"], top25["doc_freq"], color="skyblue")
ax1.set_xlabel("Share of tweets containing the term")
ax1.set_title("Top 25 terms by document frequency (Spanish tweets)")
plt.tight_layout()
st.pyplot(fig1)

# Chart 2: Top 15 TF-IDF terms per segment
st.header("2. TF-IDF Terms per Segment")
segments = list(tfidf_seg["source_type"].unique())
max_val = tfidf_seg["mean_tfidf"].max()

fig2, axes = plt.subplots(1, len(segments), figsize=(4 * len(segments), 5), sharex=True)
axes = np.atleast_1d(axes)
for ax, st_name in zip(axes, segments):
    sub = tfidf_seg[tfidf_seg["source_type"] == st_name].head(15).iloc[::-1]
    n = sub["n_docs"].iloc[0]
    ax.barh(sub["term"], sub["mean_tfidf"], color="coral")
    ax.set_title(f"{st_name} (n={n})")
    ax.set_xlim(0, max_val * 1.1)
plt.tight_layout()
st.pyplot(fig2)

# Chart 3: Monthly document frequency of key terms
st.header("3. Monthly Key Terms Trend")
df["created_at"] = pd.to_datetime(df["created_at"])
df["month"] = df["created_at"].dt.to_period("M").astype(str)
terms_track = ["agua", "energía", "empleo", "rigi", "patagonia", "inversión", "soberanía", "consumo"]

monthly = (
    df.groupby("month")["text_clean"]
      .apply(lambda s: pd.Series({t: s.str.contains(rf"\b{t}\b").mean() for t in terms_track}))
      .unstack()
)

fig3, ax3 = plt.subplots(figsize=(10, 5))
for t in terms_track:
    ax3.plot(monthly.index, monthly[t], marker="o", label=t)
ax3.set_xlabel("Month")
ax3.set_ylabel("Share of month's tweets containing the term")
ax3.set_title("Monthly document frequency of key terms")
ax3.legend(ncol=2)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)
