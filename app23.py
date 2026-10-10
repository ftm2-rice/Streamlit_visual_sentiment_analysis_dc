# ---------------------------------------------------------------------------
# Data centers in Argentina — from a raw post to the mean TF-IDF per segment.
# The app re-runs the notebook's vectorizers on posts_steps.csv(.gz) (notebook Step 2b)
# so every intermediate number can be shown. Parameters below are the notebook's.
# ---------------------------------------------------------------------------
import html
import os
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.set_page_config(page_title="TF-IDF, step by step", layout="wide")
st.title("Data centers in Argentina — how we get to the mean TF-IDF")

POSTS_FILES = ["posts_steps.csv.gz", "posts_steps.csv"]   # written by notebook Step 2b
SEG_FILE = "tfidf_by_segment.csv"                          # notebook Step 5 output, used only as a cross-check
SEGMENTS = ["outlet_post", "news_reply", "news_quote", "news_thread", "verified_opinion", "general_public"]
NGRAM, MIN_DF, MAX_DF = (1, 2), 3, 0.8                     # same as the notebook
SMALL_SEGMENT = 100
C_MAIN, C_POST, C_GREY = "#4E9AC7", "#E07B39", "#B8C2CC"   # bars / this post's terms / context

STAGES = [("text", "1 · raw post"),
          ("text_step1", "2 · links, @mentions, numbers removed"),
          ("text_step2", "3 · multiword names joined"),
          ("text_lemmas", "4 · lemmas (words only)"),
          ("text_clean", "5 · stop words removed")]


# ---------------- data ----------------
@st.cache_data
def load_posts():
    for f in POSTS_FILES:
        if os.path.exists(f):
            df = pd.read_csv(f, dtype={"tweet_id": str})
            for c, _ in STAGES:
                df[c] = df[c].fillna("").astype(str)
            return df.reset_index(drop=True), f
    return pd.DataFrame(), None


@st.cache_resource
def build_model(texts):
    cv = CountVectorizer(ngram_range=NGRAM, min_df=MIN_DF)
    X = cv.fit_transform(texts).tocsr()                                  # counts
    tv = TfidfVectorizer(ngram_range=NGRAM, min_df=MIN_DF, max_df=MAX_DF, sublinear_tf=True)
    Xt = tv.fit_transform(texts).tocsr()                                 # final TF-IDF weights
    return cv, X, tv, Xt


def removed_stopwords(lemmas, kept):
    """Lemmas dropped as stop words: walk the lemma list, consuming kept tokens in order."""
    kept, j, out = kept.split(), 0, []
    for w in lemmas.split():
        if j < len(kept) and w == kept[j]:
            j += 1
        else:
            out.append(w)
    return out


posts, posts_file = load_posts()
if posts.empty:
    st.error("`posts_steps.csv.gz` not found next to app.py. Run Step 2b of the notebook and upload "
             f"`outputs/posts_steps.csv.gz` to the repo. Files here: {sorted(os.listdir('.'))}")
    st.stop()

cv, X, tv, Xt = build_model(tuple(posts["text_clean"]))
N = len(posts)
seg_arr = posts["source_type"].astype(str).to_numpy()
segments = [s for s in SEGMENTS if s in set(seg_arr)] + sorted(set(seg_arr) - set(SEGMENTS))
seg_sizes = pd.Series(seg_arr).value_counts()

c_terms = cv.get_feature_names_out()
t_terms = tv.get_feature_names_out()
t_index = {t: i for i, t in enumerate(t_terms)}
idf = tv.idf_

terms = pd.DataFrame({"term": c_terms,
                      "total_count": np.asarray(X.sum(axis=0)).ravel(),
                      "doc_count": np.asarray((X > 0).sum(axis=0)).ravel()})
terms["doc_freq"] = terms["doc_count"] / N
terms["in_tfidf"] = terms["term"].isin(t_index)                         # False = dropped by max_df
terms["idf"] = terms["term"].map(lambda t: idf[t_index[t]] if t in t_index else np.nan)
for s in segments:
    terms[s] = np.asarray(X[seg_arr == s].sum(axis=0)).ravel()

seg_mean = pd.DataFrame({s: np.asarray(Xt[seg_arr == s].mean(axis=0)).ravel() for s in segments}, index=t_terms)


def show(term):
    return term.replace("_", " ")


def hbar(df, x, y, title, xlabel, highlight=None, height=None, hover=None):
    d = df.copy()
    d["label"] = d[y].map(show)
    d["colour"] = np.where(d[y].isin(highlight or []), "in the example post", "other terms")
    fig = px.bar(d, x=x, y="label", orientation="h", title=title, color="colour",
                 color_discrete_map={"in the example post": C_POST, "other terms": C_MAIN},
                 labels={x: xlabel, "label": "", "colour": ""}, hover_data=hover)
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=height or max(380, 24 * len(d)),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                      margin=dict(l=10, r=20, t=60, b=40))
    if not highlight:
        fig.update_layout(showlegend=False)
    return fig


# ---------------- sidebar: choose the example post ----------------
st.sidebar.header("Example post")
seg_pick = st.sidebar.selectbox("Segment", ["any"] + segments)
contains = st.sidebar.text_input("Post contains the word", "agua",
                                 help="A lemma, lowercase: agua, energía, empleo, rigi…").strip().lower()
n_clean = posts["text_clean"].str.split().str.len()
mask = pd.Series(True, index=posts.index)
if seg_pick != "any":
    mask &= posts["source_type"] == seg_pick
if contains:
    mask &= posts["text_clean"].str.split().apply(lambda toks: contains in toks)
cand = posts[mask].copy()
if cand.empty:
    st.sidebar.warning("No post matches; showing all posts.")
    cand = posts.copy()
cand["fit"] = (n_clean[cand.index] - 18).abs()                          # readable examples first (~18 terms)
cand = cand.sort_values("fit").head(60)
post_i = st.sidebar.selectbox(
    f"Post ({int(mask.sum())} match)", cand.index.tolist(),
    format_func=lambda i: f"[{posts.at[i, 'source_type']}] {posts.at[i, 'text'][:70]}…")
post = posts.loc[post_i]

st.sidebar.header("Charts")
top_n = st.sidebar.slider("Top N terms", 10, 50, 20, step=5)

# ---------------- roadmap ----------------
st.markdown(
    f"""
We follow **one post** from its raw text to its TF-IDF vector, then see how {N:,} such vectors become
the **mean TF-IDF per segment**. Pick another post in the sidebar to repeat the story.

| part | step | what happens |
|---|---|---|
| A · text | 1 – 3 | clean → lemmatize → remove stop words |
| B · counting | 4 – 6 | count terms in the post, then in the whole corpus (occurrences, posts) |
| C · weighting | 7 – 10 | IDF → damped TF → TF × IDF → normalise to length 1 |
| D · segments | 11 – 13 | average the vectors inside each segment → ranking → heatmap |
| next | sentiment | is each post positive, negative or neutral about data centers? |
"""
)

# ================= PART A · TEXT =================
st.header("Part A · From raw text to terms")

# ---- step 1 ----
st.subheader("Step 1 · The raw post")
c1, c2 = st.columns([3, 1])
c1.info(post["text"])
c2.markdown(f"**segment** `{post['source_type']}`  \n**date** {str(post.get('created_at', ''))[:10]}  \n"
            f"**tweet_id** `{post['tweet_id']}`")

# ---- step 2 ----
st.subheader("Step 2 · Cleaning: links, @mentions, numbers, multiword names")
st.markdown("Regular expressions remove what carries no meaning for this analysis, and names such as "
            "*Vaca Muerta* or *inteligencia artificial* are glued with `_` so they survive as one term.")
c1, c2 = st.columns(2)
c1.markdown("**After removing links, @mentions and numbers**")
c1.code(post["text_step1"], language=None, wrap_lines=True)
c2.markdown("**After joining multiword names**")
c2.code(post["text_step2"], language=None, wrap_lines=True)

stage_post = pd.DataFrame({"stage": [lbl for _, lbl in STAGES],
                           "words": [len(post[c].split()) for c, _ in STAGES]})
stage_all = pd.DataFrame({"stage": [lbl for _, lbl in STAGES],
                          "words": [int(posts[c].str.split().str.len().sum()) for c, _ in STAGES]})
c1, c2 = st.columns(2)
for col, d, ttl in [(c1, stage_post, "This post"), (c2, stage_all, f"All {N:,} posts")]:
    f = px.bar(d, x="words", y="stage", orientation="h", title=f"{ttl}: words left after each stage",
               color_discrete_sequence=[C_MAIN], labels={"stage": "", "words": "words"})
    f.update_layout(yaxis={"categoryorder": "array", "categoryarray": d["stage"][::-1]}, height=300,
                    margin=dict(l=10, r=20, t=50, b=40))
    col.plotly_chart(f, width="stretch")
kept_share = stage_all["words"].iloc[-1] / max(stage_all["words"].iloc[0], 1)
st.caption(f"Across the corpus, {kept_share:.0%} of the raw words survive all five stages. "
           "Stages 4 and 5 are explained in step 3.")

# ---- step 3 ----
st.subheader("Step 3 · Lemmatization and stop words")
st.markdown("spaCy (`es_core_news_sm`) turns each word into its dictionary form (*empresas → empresa*, "
            "*construyendo → construir*). Then function words (*de, que, para*) and the search terms "
            "themselves (*data center, Argentina*) are dropped. "
            "<span style='color:#888;text-decoration:line-through'>Struck-through</span> lemmas are removed.",
            unsafe_allow_html=True)
removed = removed_stopwords(post["text_lemmas"], post["text_clean"])
rem_left = Counter(removed)
chips = []
for w in post["text_lemmas"].split():
    if rem_left.get(w, 0) > 0:
        rem_left[w] -= 1
        chips.append(f"<span style='color:#888;text-decoration:line-through'>{html.escape(show(w))}</span>")
    else:
        chips.append(f"<b>{html.escape(show(w))}</b>")
st.markdown(" ".join(chips), unsafe_allow_html=True)


@st.cache_data
def corpus_removed(lemmas, kept):
    c = Counter()
    for a, b in zip(lemmas, kept):
        c.update(removed_stopwords(a, b))
    return pd.DataFrame(c.most_common(), columns=["word", "times removed"])


rem_df = corpus_removed(tuple(posts["text_lemmas"]), tuple(posts["text_clean"]))
f = px.bar(rem_df.head(top_n), x="times removed", y="word", orientation="h",
           title=f"Most frequently removed words across all posts (top {top_n})",
           color_discrete_sequence=[C_GREY], labels={"word": ""})
f.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(380, 22 * top_n))
st.plotly_chart(f, width="stretch")
st.caption("What is left (bold above) is `text_clean`, the only input to everything that follows.")

# ================= PART B · COUNTING =================
st.header("Part B · Counting")

# ---- step 4 ----
st.subheader("Step 4 · Count the terms in this post")
st.markdown(f"A **term** is one lemma or two adjacent lemmas (bigram). Only terms that appear in at least "
            f"**{MIN_DF} posts** enter the vocabulary ({len(c_terms):,} terms). The post becomes a row of counts.")
row = X[post_i]
post_counts = pd.DataFrame({"term": c_terms[row.indices], "count": row.data}).sort_values("count", ascending=False)
post_terms = post_counts["term"].tolist()
if post_counts.empty:
    st.warning("None of this post's terms reach the vocabulary. Pick another post.")
    st.stop()
st.plotly_chart(hbar(post_counts, "count", "term", f"This post: {len(post_counts)} vocabulary terms",
                     "occurrences in the post"), width="stretch")
uni_in_post = set(post["text_clean"].split())
dropped = sorted(w for w in uni_in_post if w not in cv.vocabulary_)
if dropped:
    st.caption(f"Left out because they appear in fewer than {MIN_DF} posts (or have one letter): "
               + ", ".join(show(w) for w in dropped))

# ---- step 5 ----
st.subheader("Step 5 · Total count across the corpus")
st.markdown("Add up the count rows of every post: how many times each term is used in total. "
            "A post that repeats a word three times adds 3. Orange bars are terms of the example post.")
tc = terms.sort_values("total_count", ascending=False).head(top_n)
st.plotly_chart(hbar(tc, "total_count", "term", f"Top {top_n} terms by total occurrences",
                     "total occurrences", highlight=post_terms), width="stretch")

# ---- step 6 ----
st.subheader("Step 6 · Document count: in how many posts")
st.markdown("**What changes from step 5:** each post counts once, so one long or insistent post can't "
            "inflate a word. This is the number the IDF needs.")
dc = terms.sort_values("doc_count", ascending=False).head(top_n)
st.plotly_chart(hbar(dc, "doc_count", "term", f"Top {top_n} terms by number of posts (out of {N:,})",
                     "posts mentioning the term", highlight=post_terms, hover={"doc_freq": ":.1%"}),
                width="stretch")
with st.expander("Which terms does the total count inflate?"):
    cmp = terms[terms["doc_count"] >= 10].assign(
        per_post=lambda d: (d["total_count"] / d["doc_count"]).round(2)).sort_values("per_post", ascending=False)
    st.dataframe(cmp[["term", "total_count", "doc_count", "per_post"]].head(top_n).rename(
        columns={"total_count": "total occurrences", "doc_count": "posts", "per_post": "occurrences per post"}),
        width="stretch", hide_index=True)

# ================= PART C · WEIGHTING =================
st.header("Part C · Weighting")

# ---- step 7 ----
st.subheader("Step 7 · IDF: rare terms weigh more")
st.markdown(
    rf"""
The weakness of steps 5–6: the most common words (*energía*, *ia*) always win, though they say little.
**IDF** (inverse document frequency) gives each term one corpus-wide weight, high when few posts use it:

$$\text{{idf}}(t) = \ln\frac{{1 + N}}{{1 + \text{{df}}(t)}} + 1 \qquad N = {N:,}$$

(scikit-learn's smoothed version; the $+1$ keeps every weight above zero). Terms in more than
{MAX_DF:.0%} of posts are dropped entirely (`max_df`).
"""
)
sc = terms[terms["in_tfidf"]].copy()
sc["label"] = np.where(sc["term"].isin(post_terms), sc["term"].map(show), "")
sc["colour"] = np.where(sc["term"].isin(post_terms), "in the example post", "other terms")
f = px.scatter(sc, x="doc_count", y="idf", color="colour", text="label", log_x=True,
               color_discrete_map={"in the example post": C_POST, "other terms": C_GREY},
               hover_data={"term": True, "label": False, "colour": False},
               labels={"doc_count": "posts containing the term (log scale)", "idf": "IDF", "colour": ""},
               title=f"Every term's IDF ({len(sc):,} terms): the curve is the formula")
f.update_traces(textposition="top center", marker=dict(size=6))
f.update_layout(height=480, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
st.plotly_chart(f, width="stretch")
ex_term = post_counts.iloc[0]["term"]
ex_df = int(terms.loc[terms["term"] == ex_term, "doc_count"].iloc[0])
if ex_term in t_index:
    st.caption(f"Example: *{show(ex_term)}* is in {ex_df:,} posts → idf = ln({N + 1:,} / {ex_df + 1:,}) + 1 "
               f"= {idf[t_index[ex_term]]:.3f}")

# ---- steps 8–10 for the example post ----
calc = post_counts[post_counts["term"].isin(t_index)].copy()
calc["tf"] = 1 + np.log(calc["count"])
calc["idf"] = calc["term"].map(lambda t: idf[t_index[t]])
calc["tfidf"] = calc["tf"] * calc["idf"]
length = float(np.sqrt((calc["tfidf"] ** 2).sum()))
calc["weight"] = calc["tfidf"] / length
sk = Xt[post_i]
sk_w = dict(zip(t_terms[sk.indices], sk.data))
match = np.allclose(calc["weight"], calc["term"].map(sk_w).fillna(0))

st.subheader("Step 8 · Damped term frequency (TF)")
st.markdown("Back to the example post. Its raw counts are damped with **tf = 1 + ln(count)**, so saying a "
            "word 4 times weighs 2.4×, not 4× (`sublinear_tf=True`). A single mention stays at 1.")
long8 = calc.melt(id_vars="term", value_vars=["count", "tf"], var_name="measure", value_name="value")
long8["label"] = long8["term"].map(show)
long8["measure"] = long8["measure"].map({"count": "raw count", "tf": "damped TF"})
f = px.bar(long8, x="value", y="label", color="measure", barmode="group", orientation="h",
           color_discrete_map={"raw count": C_GREY, "damped TF": C_MAIN}, labels={"label": "", "value": "", "measure": ""},
           title="Raw count vs damped TF")
f.update_layout(yaxis={"categoryorder": "array", "categoryarray": calc.sort_values("tf")["term"].map(show)},
                height=max(380, 30 * len(calc)))
st.plotly_chart(f, width="stretch")

st.subheader("Step 9 · TF × IDF")
st.markdown("Multiply each term's TF in the post by its corpus-wide IDF. Common words shrink; distinctive ones grow.")
st.plotly_chart(hbar(calc, "tfidf", "term", "TF × IDF for the terms of this post", "TF × IDF",
                     hover={"tf": ":.2f", "idf": ":.3f"}), width="stretch")

st.subheader("Step 10 · Normalise the post to length 1 (L2)")
st.markdown(f"Long posts would get bigger numbers just for being long. Each weight is divided by the "
            f"vector's length √(Σ (tf×idf)²) = **{length:.3f}**, so every post ends with length 1 "
            f"(the squares of its weights add up to 1). These are the post's final TF-IDF weights.")
st.plotly_chart(hbar(calc, "weight", "term", "Final TF-IDF weights of this post", "TF-IDF weight"),
                width="stretch")
tbl = calc[["term", "count", "tf", "idf", "tfidf", "weight"]].sort_values("weight", ascending=False).copy()
tbl["term"] = tbl["term"].map(show)
st.dataframe(tbl.rename(columns={"count": "count (4)", "tf": "tf (8)", "idf": "idf (7)", "tfidf": "tf×idf (9)",
                                 "weight": "weight (10)"}).round(3), width="stretch", hide_index=True)
st.caption(f"Sum of squared weights = {(calc['weight'] ** 2).sum():.3f}. "
           + ("✓ Identical to scikit-learn's `TfidfVectorizer` output for this post."
              if match else "⚠ Differs from scikit-learn's output; check the parameters."))

# ================= PART D · SEGMENTS =================
st.header("Part D · From posts to segments")

# ---- step 11 ----
st.subheader("Step 11 · Mean TF-IDF of one term inside each segment")
default_term = calc.sort_values("weight", ascending=False)["term"].iloc[0]
opts = sorted(t_terms, key=show)
term = st.selectbox("Term", opts, index=opts.index(default_term), format_func=show,
                    help="Defaults to the example post's heaviest term.")
j = t_index[term]
col = Xt[:, j].toarray().ravel()
dec = []
for s in segments:
    w = col[seg_arr == s]
    used = w[w > 0]
    dec.append({"segment": s, "posts": len(w), "posts using it": len(used),
                "share using it": len(used) / len(w),
                "avg weight when used": used.mean() if len(used) else 0.0,
                "mean TF-IDF": w.mean()})
dec = pd.DataFrame(dec)
st.markdown(f"Each segment's mean averages the weight of *{show(term)}* over **all** its posts, counting 0 "
            "where the post doesn't use it. So the mean is two things multiplied: "
            "**how many posts use the term × how heavily they use it**.")
long11 = dec.melt(id_vars="segment", value_vars=["share using it", "avg weight when used", "mean TF-IDF"],
                  var_name="measure", value_name="value")
f = px.bar(long11, x="segment", y="value", facet_col="measure", color="measure",
           color_discrete_map={"share using it": C_GREY, "avg weight when used": C_GREY, "mean TF-IDF": C_MAIN},
           category_orders={"segment": segments,
                            "measure": ["share using it", "avg weight when used", "mean TF-IDF"]},
           labels={"segment": "", "value": ""}, title=f"“{show(term)}”: share × weight = mean")
f.update_yaxes(matches=None, showticklabels=True)
f.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
f.update_layout(showlegend=False, height=420)
st.plotly_chart(f, width="stretch")
d = dec.copy()
d["share using it"] = (d["share using it"] * 100).round(1)
st.dataframe(d.rename(columns={"share using it": "% using it"}).round(4), width="stretch", hide_index=True)

# ---- step 12 ----
st.subheader("Step 12 · Rank every term by its mean, per segment")
st.markdown("Repeat step 11 for all terms and sort. The top of each list is what characterises that audience.")
seg_rank = st.selectbox("Segment", segments, key="rank_seg")
n_seg = int(seg_sizes.get(seg_rank, 0))
if n_seg < SMALL_SEGMENT:
    st.warning(f"`{seg_rank}` has only {n_seg} posts — its ranking is sensitive to single posts.")
rk = seg_mean[seg_rank].sort_values(ascending=False).head(top_n).rename("mean").reset_index()
rk.columns = ["term", "mean"]
st.plotly_chart(hbar(rk, "mean", "term", f"{seg_rank} (n = {n_seg:,} posts): top {top_n} by mean TF-IDF",
                     "mean TF-IDF", highlight=post_terms, hover={"mean": ":.4f"}), width="stretch")

# ---- step 13 ----
st.subheader("Step 13 · All segments side by side")
st.markdown("Terms that are in the top of any segment, with their mean in every segment: a bright row is "
            "shared by all audiences, a single bright cell belongs to one.")
top_any = set()
for s in segments:
    top_any |= set(seg_mean[s].nlargest(max(3, top_n // len(segments) + 2)).index)
heat = seg_mean.loc[sorted(top_any, key=lambda t: -seg_mean.loc[t].max()), segments]
heat.index = [show(t) for t in heat.index]
f = px.imshow(heat, color_continuous_scale="Blues", aspect="auto",
              labels=dict(x="segment", y="term", color="mean TF-IDF"))
f.update_layout(height=max(420, 22 * len(heat)))
st.plotly_chart(f, width="stretch")

# ---------------- reference ----------------
st.header("Reference")
with st.expander("Posts per segment"):
    st.dataframe(seg_sizes.reindex(segments).rename("posts").reset_index().rename(columns={"index": "segment"}),
                 width="stretch", hide_index=True)
with st.expander("Look up any term"):
    q = st.text_input("Filter terms", "")
    v = terms[terms["term"].str.replace("_", " ").str.contains(q, case=False, na=False, regex=False)] if q \
        else terms.sort_values("doc_count", ascending=False).head(top_n)
    v = v.assign(term=v["term"].map(show), doc_freq=(v["doc_freq"] * 100).round(1)).round({"idf": 3})
    st.dataframe(v[["term", "total_count", "doc_count", "doc_freq", "idf"] + segments].rename(columns={
        "total_count": "total occurrences", "doc_count": "posts", "doc_freq": "% of posts"}),
        width="stretch", hide_index=True)
    st.caption("Segment columns are occurrences in that segment. idf is empty for terms dropped by max_df.")
if os.path.exists(SEG_FILE):
    with st.expander("Cross-check against the notebook's tfidf_by_segment.csv"):
        ref = pd.read_csv(SEG_FILE)
        ref["recomputed"] = [seg_mean.at[t, s] if t in t_index and s in seg_mean.columns else np.nan
                             for t, s in zip(ref["term"], ref["source_type"])]
        ok = np.allclose(ref["mean_tfidf"], ref["recomputed"], equal_nan=False)
        (st.success if ok else st.warning)(
            "✓ The app reproduces every mean in the notebook's file." if ok else
            "The recomputed means differ from the notebook's file: the CSVs may come from different runs.")
        st.dataframe(ref.head(30), width="stretch", hide_index=True)
st.caption(f"Data: `{posts_file}` · {N:,} Spanish posts · CountVectorizer / TfidfVectorizer("
           f"ngram_range={NGRAM}, min_df={MIN_DF}, max_df={MAX_DF}, sublinear_tf=True), L2 norm.")
