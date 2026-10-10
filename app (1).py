import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="TF-IDF Explorer", layout="wide")
st.title("Data centers in Argentina — TF-IDF explorer")

SEG_FILE = "tfidf_by_segment.csv"
TERMS_FILE = "tf_terms.csv"
SEGMENTS = ["outlet_post", "news_reply", "news_quote", "news_thread", "verified_opinion", "general_public"]
SMALL_SEGMENT = 100  # segments with fewer posts than this get a "noisy" warning


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

# ---------------- facts read from the files (so the text never goes stale) ----------------
seg_sizes = (
    segment_df.groupby("segment")["n_docs"].first().astype(int).sort_values(ascending=False)
    if "n_docs" in segment_df.columns else pd.Series(dtype=int)
)
n_total = int(seg_sizes.sum()) if len(seg_sizes) else None
per_segment_stored = int(segment_df.groupby("segment").size().max())   # terms kept per segment (20)
min_docs = int(terms_df["doc_count"].min()) if "doc_count" in terms_df.columns else None
small_segments = [s for s, n in seg_sizes.items() if n < SMALL_SEGMENT]
size_table = "\n".join(f"| `{s}` | {n:,} |" for s, n in seg_sizes.items())

# ---------------- method overview ----------------
with st.expander("How this was built — read this first", expanded=True):
    st.markdown(
        f"""
**Data.** Public posts from X (Twitter) that mention data centers *and* Argentina, collected daily
with a paid search API into a Supabase database. Each post is tagged with a **segment** describing
who is speaking (see below). Retweets, bots, near-duplicates and off-topic posts are removed before
anything is counted. Spanish-language posts only{f" — **{n_total:,} posts** in this snapshot" if n_total else ""}.

**Technique.** This page uses *classical* natural-language processing, not a large language model:

1. **Cleaning** — URLs, @mentions, numbers and hashtag symbols are stripped.
2. **Lemmatization** (spaCy `es_core_news_sm`) — every word is reduced to its dictionary form, so
   *empresas / empresa* and *construyendo / construir* count as one term. Multiword names such as
   *Vaca Muerta* or *inteligencia artificial* are protected so they stay one token.
3. **Stop-word removal** — function words (*de, que, para*) plus the search terms themselves
   (*data center, Argentina*), which appear in every post and carry no information.
4. **TF-IDF weighting** (scikit-learn) — see the definition below.

Names such as *Milei* or *Stargate* do appear as terms. They were removed from the search query as
*anchors* (a post needs more than a name to be collected), not from the text, so their presence
here is real content, not a filter failure.

An LLM is *not* needed for this step: the goal is to describe **what** each audience talks about,
and a transparent word-count method does that without a black box. The LLM stage of the project
(sentiment scoring — is each post positive, negative or neutral about data centers?) comes after this.
        """
    )

with st.expander("Glossary"):
    st.markdown(
        f"""
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
{min_docs if min_docs else 3} posts. `vaca muerta` counts as a single term.

**TF** (term frequency) — how often a term appears in one post. We use a damped version
(`1 + log(count)`) so a long post that repeats a word ten times doesn't dominate.

**IDF** (inverse document frequency) — `log(N / number of posts containing the term)`. Rare terms
get a high IDF; a word used in most posts (e.g. *energía*) gets a low one.

**TF-IDF weight** — TF × IDF, then normalised so each post's vector has length 1 (the *squares* of
a post's weights add up to 1). It is high when a term is frequent **in that post** and rare
**across all posts**. Range 0–1, no units.

**Mean TF-IDF (the x-axis of chart 1, the colour of chart 2)** — the average TF-IDF weight of a
term over all posts in a segment, counting 0 for posts that don't use it. It answers: *which terms
characterise this audience compared with the corpus as a whole?* It is **not** a count, **not** the
share of posts that contain the word, and **not** a percentage. Read it as a ranking: compare terms
and segments with each other rather than interpreting a single value on its own.

**Document frequency** (table 3) — share of posts containing the term at least once. Use it when you
want *how common*, and TF-IDF when you want *how distinctive*.

**Reading tip** — means from small segments are noisy: one long post can put a term at the top.
{("Segments under " + str(SMALL_SEGMENT) + " posts in this snapshot: " + ", ".join(f"`{s}` ({seg_sizes[s]})" for s in small_segments) + ". Trust the larger ones more.") if small_segments else ""}

**Posts per segment in this snapshot**

| segment | posts |
|---|---|
{size_table}
        """
    )

# ---------------- sidebar ----------------
st.sidebar.header("Controls")
top_n = st.sidebar.slider(
    "Top N terms", 5, per_segment_stored, min(15, per_segment_stored),
    help=f"The file stores the top {per_segment_stored} terms per segment, so that is the maximum.",
)
segments = [s for s in SEGMENTS if s in set(segment_df["segment"].astype(str))]
segments += sorted(set(segment_df["segment"].astype(str)) - set(segments))  # any unexpected segment last
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
n_docs = int(seg_sizes[selected]) if selected in seg_sizes.index else None
title = selected + (f"  (n = {n_docs:,} posts)" if n_docs else "")
if n_docs and n_docs < SMALL_SEGMENT:
    st.warning(f"`{selected}` has only {n_docs} posts — its ranking is sensitive to single posts.")

fig_bar = px.bar(
    sub,
    x="tfidf_score",
    y="term",
    orientation="h",
    color="tfidf_score",
    color_continuous_scale="Blues",
    labels={"tfidf_score": "Mean TF-IDF weight", "term": ""},
    title=title,
)
fig_bar.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=max(400, 30 * top_n),
    coloraxis_showscale=False,
)
fig_bar.update_traces(hovertemplate="%{y}: %{x:.4f}<extra></extra>")
st.plotly_chart(fig_bar, width="stretch")

# ---------------- 2. heatmap ----------------
st.header("2. Term × segment heatmap")
st.markdown("Same measure as chart 1, laid out as a grid so you can see which terms are shared "
            "across audiences (a bright row) and which belong to one audience only (a single bright cell).")
pivot = segment_df.pivot_table(index="term", columns="segment", values="tfidf_score", fill_value=0)
pivot = pivot[[s for s in segments if s in pivot.columns]]          # same column order as the selector
top_terms = segment_df.groupby("term")["tfidf_score"].sum().nlargest(top_n).index
heat = pivot.loc[pivot.index.isin(top_terms)]
heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]  # strongest terms at the top
fig_heat = px.imshow(
    heat,
    labels=dict(x="Segment", y="Term", color="Mean TF-IDF"),
    color_continuous_scale="Viridis",
    aspect="auto",
    title=f"Top {top_n} terms across segments",
)
fig_heat.update_layout(height=max(400, 26 * top_n))
st.plotly_chart(fig_heat, width="stretch")
st.caption(f"Only each segment's top {per_segment_stored} terms are stored, so a 0 cell means "
           "\"not in that segment's top list\", not necessarily \"never used\".")

# ---------------- 3. lookup ----------------
st.header("3. Term frequency lookup")
st.markdown("Raw counts, independent of TF-IDF. `posts mentioning` = posts containing the term, "
            "`% of posts` = share of all posts, `total occurrences` = every use of the term. "
            "The segment columns are **occurrences** in that segment (they add up to `total occurrences`), "
            "not numbers of posts.")
query = st.text_input("Filter terms", "")
if terms_df.empty:
    st.info(f"`{TERMS_FILE}` not found — lookup disabled.")
else:
    view = (
        terms_df[terms_df["term"].str.contains(query, case=False, na=False, regex=False)]
        if query
        else terms_df.sort_values("doc_count", ascending=False).head(top_n)
    ).copy()
    seg_cols = [s for s in segments if s in view.columns]
    if "doc_freq" in view.columns:
        view["doc_freq"] = (view["doc_freq"] * 100).round(1)
    view = view[[c for c in ["term", "doc_count", "doc_freq", "total_count"] if c in view.columns] + seg_cols]
    view = view.rename(columns={"doc_count": "posts mentioning", "doc_freq": "% of posts",
                                "total_count": "total occurrences"})
    st.dataframe(view, width="stretch", hide_index=True)
    st.caption(f"{len(view):,} terms shown" + (f" of {len(terms_df):,}" if query else ""))
