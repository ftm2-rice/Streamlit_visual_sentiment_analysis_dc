import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="TF-IDF Interactive Explorer", layout="wide")

st.title("📊 Interactive TF-IDF & Text Segment Explorer")

# Data loading with caching for performance
@st.cache_data
def load_data():
    segment_df = pd.read_csv('tfidf_by_segment.csv') if os.path.exists('tfidf_by_segment.csv') else pd.DataFrame()
    terms_df = pd.read_csv('tf_terms.csv') if os.path.exists('tf_terms.csv') else pd.DataFrame()
    index_df = pd.read_csv('tfidf_index.csv') if os.path.exists('tfidf_index.csv') else pd.DataFrame()
    return segment_df, terms_df, index_df

segment_df, terms_df, index_df = load_data()

# Sidebar controls
st.sidebar.header("Global Controls")
top_n = st.sidebar.slider("Top N Terms to Display", min_value=5, max_value=50, value=15)

# Section 1: Interactive Segment Analysis
st.header("1. TF-IDF Score by Segment")

if not segment_df.empty and {'segment', 'term', 'tfidf_score'}.issubset(segment_df.columns):
    segments = sorted(segment_df['segment'].astype(str).unique())
    selected_segment = st.sidebar.selectbox("Select Segment Focus", segments)

    # Filter data dynamically
    filtered_segment = (
        segment_df[segment_df['segment'].astype(str) == selected_segment]
        .sort_values(by='tfidf_score', ascending=False)
        .head(top_n)
    )

    # Plotly Interactive Horizontal Bar Chart
    fig_bar = px.bar(
        filtered_segment,
        x='tfidf_score',
        y='term',
        orientation='h',
        color='tfidf_score',
        color_continuous_scale='Blues',
        labels={'tfidf_score': 'TF-IDF Score', 'term': 'Term'},
        title=f"Top {top_n} Terms in Segment: '{selected_segment}'",
        hover_data=['tfidf_score']
    )
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.warning("`tfidf_by_segment.csv` not found or missing required columns ('segment', 'term', 'tfidf_score').")

# Section 2: Multi-Segment Comparison Heatmap
st.header("2. Segment-Term Heatmap Matrix")

if not segment_df.empty and {'segment', 'term', 'tfidf_score'}.issubset(segment_df.columns):
    # Pivot matrix for heatmap visualization
    pivot_df = segment_df.pivot_table(index='term', columns='segment', values='tfidf_score', fill_value=0)
    top_global_terms = segment_df.groupby('term')['tfidf_score'].sum().nlargest(top_n).index
    pivot_filtered = pivot_df.loc[pivot_df.index.isin(top_global_terms)]

    fig_heatmap = px.imshow(
        pivot_filtered,
        labels=dict(x="Segment", y="Term", color="TF-IDF Score"),
        x=pivot_filtered.columns.astype(str),
        y=pivot_filtered.index,
        color_continuous_scale='Viridis',
        title=f"TF-IDF Distribution Across Segments (Top {top_n} Global Terms)"
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Section 3: Interactive Keyword Search & Index Inspector
st.header("3. Keyword Index & Document Lookup")

col1, col2 = st.columns([1, 2])

with col1:
    search_query = st.text_input("Filter by Keyword", value="")

with col2:
    if not terms_df.empty:
        st.subheader("Global Term Frequency Ranking")
        filtered_terms = terms_df
        if search_query:
            filtered_terms = terms_df[terms_df['term'].str.contains(search_query, case=False, na=False)]
        
        st.dataframe(filtered_terms.head(top_n), use_container_width=True)

if search_query and not index_df.empty:
    st.subheader(f"Document Index Matches for '{search_query}'")
    matches = index_df[index_df['term'].str.contains(search_query, case=False, na=False)] if 'term' in index_df.columns else index_df
    st.dataframe(matches, use_container_width=True)
