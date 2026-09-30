# ---------------------------------------------------------------------------
# Streamlit app: "Which words appear most in tweets about data centers in Argentina?"
# One chart: top terms by share of posts (document frequency), read from tf_terms.csv.
# ---------------------------------------------------------------------------

import os                      # to check whether the CSV file exists and list the folder
import streamlit as st         # the web-app framework (every st.* call draws something on the page)
import pandas as pd            # tables (DataFrames)
import plotly.express as px    # interactive charts

# Page-level settings: browser tab title and use the full width of the window.
st.set_page_config(page_title="Term frequency — data centers in Argentina", layout="wide")

# Big title at the top of the page.
st.title("Which words appear most in tweets about data centers in Argentina?")

# Name of the input file. It must sit in the same folder as app.py.
# It is produced by the notebook cell "# 3. Term frequency" (tf_terms.to_csv(...)).
TERMS_FILE = "tf_terms.csv"


@st.cache_data                 # remember the result so the CSV is read once, not on every click
def load_terms():
    if not os.path.exists(TERMS_FILE):          # file missing -> return an empty table
        return pd.DataFrame()
    df = pd.read_csv(TERMS_FILE)                # read the CSV into a DataFrame
    df["term"] = df["term"].astype(str).str.replace("_", " ")   # "vaca_muerta" -> "vaca muerta" for display
    return df


terms_df = load_terms()        # run the loader; terms_df now holds the table (or is empty)

# If the file is missing, show a red error with the folder contents (helps debugging on Streamlit Cloud) and stop.
if terms_df.empty:
    st.error(f"`{TERMS_FILE}` not found next to app.py. Files here: {sorted(os.listdir('.'))}")
    st.stop()

# If the file exists but lacks the two columns the chart needs, say which columns it has, and stop.
if not {"term", "doc_freq"}.issubset(terms_df.columns):
    st.error(f"`{TERMS_FILE}` has columns {list(terms_df.columns)}; expected at least `term` and `doc_freq`.")
    st.stop()

# ---------------- explanation ----------------
# A collapsible box (open by default) with the method explanation, written in Markdown.
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
st.sidebar.header("Controls")                                      # heading in the left sidebar
top_n = st.sidebar.slider("Number of terms", 10, 60, 25, step=5)   # slider: min 10, max 60, default 25
ngram = st.sidebar.radio("Show", ["All terms", "Single words only", "Two-word terms only"])  # radio buttons

data = terms_df.copy()                          # work on a copy so the cached original stays untouched
n_words = data["term"].str.count(" ") + 1       # number of words in each term = spaces + 1
if ngram == "Single words only":
    data = data[n_words == 1]                   # keep unigrams only
elif ngram == "Two-word terms only":
    data = data[n_words == 2]                   # keep bigrams only
# ("All terms" keeps everything)

# Sort by share of posts, highest first, and keep the top N chosen in the slider.
top = data.sort_values("doc_freq", ascending=False).head(top_n)

# ---------------- chart ----------------
fig = px.bar(
    top,                                        # the data
    x=top["doc_freq"] * 100,                    # x = share of posts, converted from 0–1 to 0–100 (%)
    y="term",                                   # y = the term (one bar per term)
    orientation="h",                            # horizontal bars
    color_discrete_sequence=["#4E9AC7"],        # single bar colour
    labels={"x": "% of posts containing the term", "term": ""},   # axis titles
    title=f"Top {top_n} terms by share of posts",
    hover_data={"doc_count": True} if "doc_count" in top.columns else None,  # show raw count on hover
)
fig.update_layout(
    yaxis={"categoryorder": "total ascending"}, # longest bar at the top
    height=max(450, 24 * top_n),                # taller chart when more terms are shown
    xaxis_ticksuffix="%",                       # put a % sign on the x-axis ticks
    margin=dict(l=10, r=20, t=50, b=40),        # trim empty space around the plot
)
fig.update_traces(hovertemplate="%{y}: %{x:.1f}% of posts<extra></extra>")  # tooltip text
st.plotly_chart(fig, width="stretch")          # draw the chart, full width

# Work out the corpus size from the file itself: doc_count / doc_freq = total number of posts.
n_total = None
if "doc_count" in terms_df.columns and "doc_freq" in terms_df.columns:
    first = terms_df.iloc[0]                    # any row works; take the first
    if first["doc_freq"] > 0:
        n_total = int(round(first["doc_count"] / first["doc_freq"]))
if n_total:
    st.caption(f"Corpus: {n_total} Spanish-language posts. A term is included if it appears in at least 3 posts.")

# ---------------- table ----------------
# A collapsed box with the numbers behind the bars.
with st.expander("Table"):
    cols = [c for c in ["term", "doc_count", "doc_freq", "total_count"] if c in top.columns]  # columns that exist
    view = top[cols].copy()
    view["doc_freq"] = (view["doc_freq"] * 100).round(1)          # show share as a percentage with 1 decimal
    view = view.rename(columns={"doc_count": "posts mentioning",  # friendlier column names
                                "doc_freq": "% of posts",
                                "total_count": "total occurrences"})
    st.dataframe(view, width="stretch", hide_index=True)          # render the table without the index column
