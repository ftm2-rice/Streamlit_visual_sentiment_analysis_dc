import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Term frequency — data centers in Argentina", layout="wide")
st.title("Which words appear most in tweets about data centers in Argentina?")

TERMS_FILE = "tf_terms.csv"


@st.cache_data
def load_terms():
    if not os.path.exists(TERMS_FILE):
        return pd.DataFrame()
    df = pd.read_csv(TERMS_FILE)
    df["term"] = df["term"].astype(str).str.replace("_", " ")
    return df


terms_df = load_terms()

if terms_df.empty:
    st.error(f"`{TERMS_FILE}` not found next to app.py. Files here: {sorted(os.listdir('.'))}")
    st.stop()
if not {"term", "doc_freq"}.issubset(terms_df.columns):
    st.error(f"`{TERMS_FILE}` has columns {list(terms_df.columns)}; expected at least `term` and `doc_freq`.")
    st.stop()

# ---------------- explanation ----------------
with st.expander("What this chart shows", expanded=True):
    st.markdown(
        """
**Data.** Spanish-language posts from X (Twitter) that mention data centers *and* Argentina,
collected daily into a database. Retweets, bots, near-duplicates and off-topic posts were removed.

**Preparation.** URLs, @mentions and numbers were stripped; each word was reduced to its
dictionary form (*empresas → empresa*); function words (*de, que, para*) and the search terms
themselves (*data center, Argentina*) were dropped because they appear in every post.
Multiword names such as *vaca muerta* are kept as one term.

**Measure — document frequency.** For each term, the **share of posts that mention it at least
once**. A bar at 34 % for *energía* means 34 out of every 100 posts contain the word *energía*.
It does not matter how many times a post repeats the word: a post counts once.

This is a plain count, no weighting. It answers *what is the conversation about?*
It does not say whether the mention is positive or negative — that is the sentiment step.
        """
    )

# ---------------- controls ----------------
st.sidebar.header("Controls")
top_n = st.sidebar.slider("Number of terms", 10, 60, 25, step=5)
ngram = st.sidebar.radio("Show", ["All terms", "Single words only", "Two-word terms only"])

data = terms_df.copy()
n_words = data["term"].str.count(" ") + 1
if ngram == "Single words only":
    data = data[n_words == 1]
elif ngram == "Two-word terms only":
    data = data[n_words == 2]

top = data.sort_values("doc_freq", ascending=False).head(top_n)

# ---------------- chart ----------------
fig = px.bar(
    top,
    x=top["doc_freq"] * 100,
    y="term",
    orientation="h",
    color_discrete_sequence=["#4E9AC7"],
    labels={"x": "% of posts containing the term", "term": ""},
    title=f"Top {top_n} terms by share of posts",
    hover_data={"doc_count": True} if "doc_count" in top.columns else None,
)
fig.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=max(450, 24 * top_n),
    xaxis_ticksuffix="%",
    margin=dict(l=10, r=20, t=50, b=40),
)
fig.update_traces(hovertemplate="%{y}: %{x:.1f}% of posts<extra></extra>")
st.plotly_chart(fig, width="stretch")

n_total = None
if "doc_count" in terms_df.columns and "doc_freq" in terms_df.columns:
    first = terms_df.iloc[0]
    if first["doc_freq"] > 0:
        n_total = int(round(first["doc_count"] / first["doc_freq"]))
if n_total:
    st.caption(f"Corpus: {n_total} Spanish-language posts. A term is included if it appears in at least 3 posts.")

# ---------------- table ----------------
with st.expander("Table"):
    cols = [c for c in ["term", "doc_count", "doc_freq", "total_count"] if c in top.columns]
    view = top[cols].copy()
    view["doc_freq"] = (view["doc_freq"] * 100).round(1)
    view = view.rename(columns={"doc_count": "posts mentioning", "doc_freq": "% of posts",
                                "total_count": "total occurrences"})
    st.dataframe(view, width="stretch", hide_index=True)
