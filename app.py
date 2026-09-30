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


# ---------------- method overview ----------------
with st.expander("How this was built — read this first", expanded=True):
    st.markdown(
        """
**Data.** Public posts from X (Twitter) that mention data centers *and* Argentina, collected daily
with a paid search API into a Supabase database. Each post is tagged with a **segment** describing
who is speaking (see below). Retweets, bots, near-duplicates and off-topic posts are removed before
anything is counted. Spanish-language posts only.

**Technique.** This page uses *classical* natural-language processing, not a large language model:

1. **Cleaning** — URLs, @mentions, numbers and hashtag symbols are stripped.
2. **Lemmatization** (spaCy `es_core_news_sm`) — every word is reduced to its dictionary form, so
   *empresas / empresa* and *construyendo / construir* count as one term. Multiword names such as
   *Vaca Muerta* or *inteligencia artificial* are protected so they stay one token.
3. **Stop-word removal** — function words (*de, que, para*) plus the search terms themselves
   (*data center, Argentina*), which appear in every post and carry no information.
4. **TF-IDF weighting** (scikit-learn) — see the definition below.

An LLM is *not* needed for this step: the goal is to describe **what** each audience talks about,
and a transparent word-count method does that without a black box. The LLM stage of the project
(sentiment scoring — is each post positive, negative or neutral about data centers?) comes after this.
        """
    )

with st.expander("Glossary"):
    st.markdown(
        """
**Segment** — who wrote the post, assigned by rules at collection time:

| segment | meaning |
|---|---|
| `outlet_post` | the headline post of a news outlet or official account — information, not opinion |
| `news_reply` | a direct reply to an outlet's post |
| `news_quote` | a quote-repost of an outlet's post with the person's own comment |
| `news_thread` | a reply inside an outlet's thread, addressed to another commenter |
| `verified_opinion` | a standalone post by a verified account with ≥ 2,000 followers (journalists, analysts, politicians, paid blue checks) |
| `general_public` | everyone else |

`news_reply + news_quote + news_thread` together are *reactions to news coverage*.

**Term** — a lemmatized word (unigram) or pair of adjacent words (bigram) that appears in at least
3 posts. `vaca muerta` counts as a single term.

**TF** (term frequency) — how often a term appears in one post. We use a damped version
(`1 + log(count)`) so a long post that repeats a word ten times doesn't dominate.

**IDF** (inverse document frequency) — `log(N / number of posts containing the term)`. Rare terms
get a high IDF; a word used in most posts (e.g. *energía*) gets a low one.

**TF-IDF weight** — TF × IDF, then normalised so each post's vector has length 1. It is high when a
term is frequent **in that post** and rare **across all posts**. Range 0–1, no units.

**Mean TF-IDF (the x-axis of chart 1, the colour of chart 2)** — the average TF-IDF weight of a
term over all posts in a segment. It answers: *which terms characterise this audience compared with
the corpus as a whole?* It is **not** a count and **not** the share of posts that contain the word.
A value of 0.05 means the term carries, on average, 5 % of the weight of a post in that segment.

**Document frequency** (table 3) — share of posts containing the term at least once. Use it when you
want *how common*, and TF-IDF when you want *how distinctive*.

**Reading tip** — small segments (`news_quote`, `news_reply` with a few dozen posts) have noisy
means: one long post can put a term at the top. Trust the three large segments more.
        """
    )

# ---------------- sidebar ----------------
st.sidebar.header("Controls")
top_n = st.sidebar.slider("Top N terms", 5, 50, 15)
segments = sorted(segment_df["segment"].astype(str).unique())
selected = st.sidebar.selectbox("Segment", segments)

# ---------------- 1. bar chart ----------------
st.header("1. Most characteristic terms per segment")
st.markdown("Bars show the **mean TF-IDF weight** of each term across the posts in the selected "
            "segment: longer bar = the term is both frequent within this audience and distinctive "
            "relative to everyone else. Compare segments by switching the selector on the left.")
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
st.markdown("Same measure as chart 1, laid out as a grid so you can see which terms are shared "
            "across audiences (a bright row) and which belong to one audience only (a single bright cell).")
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
st.markdown("Raw counts, independent of TF-IDF: `total_count` = occurrences, `doc_count` = posts "
            "containing the term, `doc_freq` = share of posts. The remaining columns are counts per segment.")
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
