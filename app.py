import streamlit as st
import pandas as pd
import plotly.express as px

def render_plotly_dashboard(df):
    # Render interactive Plotly chart for top terms
    st.subheader("Top Terms Interactive Distribution")
    fig = px.bar(
        df.head(15), 
        x="Term", 
        y="Frequency", 
        title="Top 15 Most Frequent Terms",
        labels={"Term": "Terms", "Frequency": "Count"}
    )
    st.plotly_chart(fig, use_container_width=True)
